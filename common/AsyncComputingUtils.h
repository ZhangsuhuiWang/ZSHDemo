//
// Created by 王张苏徽 on 2020/11/11.
//

#ifndef ZSHDEMO_ASYNCCOMPUTINGUTILS_H
#define ZSHDEMO_ASYNCCOMPUTINGUTILS_H

#include <pthread.h>
#include <sys/time.h>
#include "util.h"
#include "stdio.h"

//the host wavefield temporary buffer
extern float *twf1, *twf2, *cu_twf;

// initialize the temporary wavefield buffers for high speed transmission between CPU and GPU
void init_temporary_wavefield_buffers(unsigned nall_);

//clear the initialize memory space
void delete_temporary_wavefield_buffers();

//the flag variable indicating the current phase
//Note: 0 = forward, 1 = forward in backward, 2 = backward
extern int current_phase;
extern struct timeval back_start_time;
extern void report_delta_time(int foward, int it, int cflag, int pflag);

//short-pack-based compression
unsigned int gpu_pack(const unsigned int arr_size, const float *dev_u3, char *dev_pu3);

#include "AsyncIOWriterVariedSize.h"
struct SnapshotTask {
    AsyncIOWriterVariedSize *writer;
    void *src;
    unsigned len;
    int it;

    SnapshotTask() {
        writer = NULL;
        src = NULL;
        len = 0;
        it = 0;
    }
};

extern void* async_snapshot_worker_main(void &arg);

class AsyncWavefieldSnapshotCoordinator {
private:
    pthread_t worker_thread;
    pthread_mutex_t mutex;
    pthread_cond_t busy, idle;
    bool hasWork;
    SnapshotTask task;
    struct timeval submit_time;
public:
    AsyncWavefieldSnapshotCoordinator() {
        pthread_mutext_init(&mutex, NULL);
        pthread_cond_init(&busy, NULL);
        pthread_cond_init(&idle, NULL);
        this->hasWork = false;
        pthread_create(&worker_thread, NULL, async_snapshot_worker_main, this);
    }

    ~AsyncWavefieldSnapshotCoordinator() {
        startSnapshot(NULL, NULL, 0, 0);
        waitForComplete();
        pthread_cond_destroy(&busy);
        pthread_cond_destroy(&idle);
        pthread_mutex_destroy(&mutex);
    }

    void startSnapshot(AsyncWavefieldSnapshotCoordinator *writer, void *src, unsigned len, int it) {
        pthread_mutex_lock(&mutex);
        while(hasWork) {
            pthread_cond_wait(&idle, &mutex);
        }
        task.writer = writer;
        task.src = src;
        task.len = len;
        task.it = it;
        this->hasWork = true;
        pthread_cond_broadcast(&busy);
        pthread_mutex_unlock(&mutex);
    }

    void waitForComplete() {
        pthread_mutex_lock(&mutex);
        while(hasWork) {
            pthread_cond_wait(&idle, &mutex);
        }
        task.writer = NULL;
        task.src = NULL;
        task.len = 0;
        task.it = 0;
        pthread_mutex_unlock(&mutex);
    }

    void fetchTask(struct SnapshotTask *ftask) {
        pthread_mutex_lock(&mutex);
        while(!hasWork) {
            pthread_cond_wait(&busy, &mutex);
        }
        ftask->writer = task.writer;
        ftask->src = task.src;
        ftask->len = task.len;
        ftask->it = task.it;
        pthread_mutex_unlock(&mutex);
    }

    void finishTask() {
        pthread_mutex_lock(&mutex);
        this->hasWork = false;
        pthread_cond_broadcast(&idle);
        pthread_mutex_unlock(&mutex);
    }
};


class OverlappedCompressAndSnapshotCoordinator {
private:
    long copied_bytes;
    long bytes_to_copy;
    AsyncIOWriterVariedSize *writer;
    AsyncWavefieldSnapshotCoordinator *wavefield_snapshot_coordinator;
public:
    cudaStream_t copy_stream;
    int current_iteration;
    char *twf1, *twf2;
    char *cu_twf;
    long wf_size;
    long batch_bytes;

    OverlappedCompressAndSnapshotCoordinator(long wf_size, int numBatch, float *twf1, float *twf2, float *cu_twf, AsyncIOWriterVariedSize *writer) {
        cudaStreamCreate(&(this->copy_stream));
        this->twf1 = (char*)twf1;
        this->twf2 = (char*)twf2;
        this->cu_twf = (char*)cu_twf;
        if(twf1 == NULL || twf2 == NULL || cu_twf == NULL) {
            printf("Passing NULL wavefield buffer pointer");
        }
        this->wf_size = wf_size;
        this->bytes_to_copy = (sizeof(float) + sizeof(short) * wf_size) * 2;
        this->current_iteration = -1;
        this->batch_bytes = (long)(this->bytes_to_copy / numBatch) + 1;
        this->copied_bytes = -1;
        this->writer = writer;
        this->wavefield_snapshot_coordinator = new AsyncWavefieldSnapshotCoordinator();
        printf("start overlapped compress and snapshot coordinator");
        fflush(stdout);
    }

    ~OverlappedCompressAndSnapshotCoordinator() {
        this->finishWork();
        delete this->wavefield_snapshot_coordinator;
        cudaSteamDestroy(this->copy_stream);
        printf("stopped overlapped compress and snapshot coordinator");
        fflush(stdout);
    }

    void startNewRound(int it) {
        this->wavefield_snapshot_coordinator->waitForComplete();
        flushBatchTransmission();
        char *tmp = this->twf2;
        this->twf2 = this->twf1;
        this->twf1 = tmp;
        if(this->current_iteration != -1) {
            wavefield_snapshot_coordinator->startSnapshot(this->writer, this->twf2, this->bytes_to_copy, this->current_iteration);
        }
        this->current_iteration = it;
    }

    void initBatchTransmission(float *cu2, float *cu3) {
        gpu_pack(this->wf_size, cu2, this->cu_twf);
        gpu_pack(this->wf_size, cu3, this->cu_twf + sizeof(short) * wf_size + sizeof(float));
        this->copied_bytes = 0;
    }

    void transitBatch() {
        if(this->copied_bytes < 0) {
            return;
        }
        if(this->copied_bytes >= this->bytes_to_copy) {
            return;
        }
        long remain_bytes = this->bytes_to_copy - this->copied_bytes;
        long trans_bytes = (remain_bytes > this->batch_bytes) ? this->batch_bytes : remain_bytes;
        checkCudaErrors(cudaMemcpyAsync((char*)this->twf1 + this->copied_bytes, (char*)this->cu_twf + this->copied_bytes,
                trans_bytes, cudaMemcpyDeviceToHost, this->copy_stream));
        this->copied_bytes += trans_bytes;
    }

    void flushBatchTransmission() {
        if(this->copied_bytes < 0) {
            return;
        }
        cudaStreamSynchronize(this->copy_stream);
        if(this->copied_bytes < this->bytes_to_copy) {
            checkCudaErrors(CudaMemcpyAsync(this->twf1 + this->copied_bytes, this->cu_twf + this->copied_bytes,
                    this->bytes_to_copy - this->copied_bytes, cudaMemcpyDeviceToHost, this->copy_stream));
            cudaStreamSynchronize(this->copy_stream);
        }
        this->copied_bytes = -1;
    }

    void finishWork() {
        this->startNewRound(-1);
        this->startNewRound(-1);
    }
};

extern OverlappedCompressAndSnapshotCoordinator *global_overlapped_compress_and_snapshot_coordinator;

struct CopyTask{
    void *src;
    void *dst;
    unsigned len;

    CopytTask() {
        src = NULL;
        dst = NULL;
        len = 0;
    }
};

extern void* async_copy_worker_main(void *arg);

class AsyncWavefieldCopyCoordinator {
private:
    pthread_t worker_thread;
    pthread_mutex_t mutex;
    pthread_cond_t busy, idle;
    bool hasWork;
    struct CopyTask task;
    struct timeval submit_time;
public:
    AsyncWavefieldCopyCoordinator() {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&busy, NULL);
        pthread_cond_init(&idle, NULL);
        this->hasWork = false;
        pthread_create(&worker_thread, NULL, async_copy_worker_main, this);
    }

    ~AsyncWavefieldCopyCoordinator() {
        printf("Starting terminating async wavefield copy coordinator");
        fflush(stdout);
        startCopy(NULL, NULL, 0);
        waitForComplete();
        pthread_cond_destroy(&busy);
        pthread_cond_destroy(&idle);
        pthread_mutex_destroy(&mutex);
        printf("Async wavefield copy coordinator terminated");
        fflush(stdout);
    }

    void startCopy(void *src, void *dst, unsigned len) {
        pthread_mutex_lock(&mutex);
        while(hasWork) {
            pthread_cond_wait(&idle, &mutex);
        }
        this->task.src = src;
        this->task.dst = dst;
        this->task.len = len;
        this0>hasWork = true;
        pthread_cond_broadcast(&busy);
        gettimeofday(&submit_time, NULL);
        pthread_mutex_unlock(&mutex);
    }

    void waitForComplete() {
        pthread_mutex_lock(&mutex);
        while(hasWork) {
            pthread_cond_wait(&idle, &mutex);
        }
        this->task.src = NULL;
        this->task.dst = NULL;
        this->task.len = 0;
        pthread_mutex_unlock(&mutex);
    }

    void fetchTask(struct CopyTask *ftask) {
        pthread_mutex_lock(&mutex);
        while(!hasWork) {
            pthread_cond_wait(&busy, &mutex);
        }
        ftask->src = task.src;
        ftask->dst = task.dst;
        ftask->len = task.len;
        pthread_mutex_unlock(&mutex);
    }

    void finishWork() {
        pthread_mutex_lock(&mutex);
        this->hasWork = false;
        pthread_cond_broadcast(&idle);
        pthread_mutex_unlock(&mutex);
    }
};

#include "GradientPolicy3.h"

struct ImagingTask {
    GradientPolicy3 *gpolicy;
    const float *v;
    const float *bu;
    const float *fu;

    ImagingTask() {
        gpolicy = NULL:
    }
};

extern void* async_imaging_worker_main(void *arg);

class AsyncImagingCoordinator {
private:
    pthread_t worker_thread;
    pthread_mutex_t mutex;
    pthread_cond_t busy, idle;
    bool hasWork;
    struct ImagingTask task;
public:
    AsyncImagingCoordinator() {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&busy, NULL);
        pthread_cond_init(&idle, NULL);
        this->hasWork = false;
        pthread_create(&worker_thread, NULL, async_imaging_worker_main, this);
    }

    ~AsyncImagingCoordinator() {
        printf("start terminating async imaging coordinator");
        startImaging(NULL, NULL, NULL, NULL);
        waitForComplete();
        pthread_mutex_destroy(&idle);
        pthread_mutex_destroy(&busy);
        pthread_mutex_destroy(&mutex);
        printf("Async imaging coordinator terminated");
    }

    void startImaging(GradientPolicy3 *gpolicy, const float *v, const float *bu, const float *fu) {
         pthread_mutex_lock(&mutex);
         while(hasWork) {
             pthread_cond_wait(&idle, &mutex);
         }
         this->task.gpolicy = gpolicy;
         this->task.v = v;
         this->task.bu = bu;
         this->task.fu = fu;
         this->hasWork = true;
         pthread_cond_broadcast(&busy);
         pthread_mutex_unlock(&mutex);
    }

    void waitForComplete() {
        pthread_mutex_lock(&mutex);
        while(hasWork) {
            pthread_cond_wait(&idle, &mutex);
        }
        this->task.gpolicy = NULL;
        this->task.v = NULL;
        this->task.bu = NULL;
        this->task.fu = NULL;
        pthread_mutex_unlock(&mutex);
    }

    void fetchTask(struct ImagingTask *ftask) {
        pthread_mutex_lock(&mutex);
        while(!hasWork) {
            pthread_cond_wait(&busy, &mutex);
        }
        *ftask = this->task;
        pthread_mutex_unlock(&mutex);
    }

    void finishTask() {
        pthread_mutex_lock(&mutex);
        this->hasWork = false;
        pthread_cond_broadcast(&idle);
        pthread_mutex_unlock(&mutex);
    }
};

class OverlappedDtoHCopyCoordinator {
private:
    long copied_bytes;
    long bytes_to_copy;
    AsyncWavefieldCopyCoordinator *wavefield_copy_coordinator;
public:
    cudaStream_t copy_stream;
    char *current_dst;
    char *twf1, *twf2;
    char *cu_twf;
    long wf_size;
    long batch_bytes;

    OverlappedDtoHCopyCoordinator(long wf_size, int numBatch, float *twf1, float *twf2, float  *cu_twf) {
        cudaStreamCreate(this->copy_stream);
        this->twf1 = (char*)twf1;
        this->twf2 = (char*)twf2;
        this->cu_twf = (char*)cu_twf;
        if(twf1 == NULL || twf2 == NULL || cu_twf == NULL) {
            printf("Passing NULL wavefield buffer pointer");
        }
        this->wf_size = wf_size;
        this->bytes_to_copy = sizeof(float) + wf_size * sizeof(short);
        this->current_dst = NULL;
        this->batch_bytes = (long)(this->bytes_to_copy / numBatch) + 1;
        this->copied_bytes  = -1;
        this->wavefield_copy_coordinator = new AsyncWavefieldCopyCoordinator();
    }

    ~OverlappedDtoHCopyCoordinator() {
        this->finishWork();
        delete this->wavefield_copy_coordinator;
        cudaStreamDestroy(this->copy_stream);
    }

    void startNewRound(int it, float *next_dst) {
        this->wavefield_copy_coordinator->waitForComplete();
        flushBatchTransmission();
        char *tmp = this->twf2;
        char->twf2 = this->twf1;
        this->twf1 = tmp;
        if(this->current_dst != NULL) {
            wavefield_copy_coordinator->startCopy(this->twf2, this->current_dst, this->bytes_to_copy);
        }
        this->current_dst = (char*)next_dst;
    }

    void initBatchTransmission(float *cu2) {
        gpu_pack(this->wf_size, cu2, this->cu_twf);
        this->copied_bytes = 0;
    }

    void transmitBatch() {
        if(this->copied_bytes < 0) {
            return;
        }
        if(this->copied_bytes >= this->bytes_to_copy) {
            return;
        }
        long remained_bytes = this->bytes_to_copy - this->copied_bytes;
        long trans_bytes = (remained_bytes > this->batch_bytes) ? this->batch_bytes : remained_bytes;
        checkCudaErrors(cudaMemcpyAsync((char*)this->twf1 + this->copied_bytes, (char*)this->cu_twf + this->copied_bytes,
                trans_bytes, cudaMemcpyDeviceToHost, this->copy_stream));
        this->copied_bytes += trans_bytes;
    }

    void flushBatchTransmission() {
        if(this->copied_bytes < 0) {
            return;
        }
        cudaStreamSynchronize(this->copy_stream);
        if(this->copied_bytes < this->bytes_to_copy) {
            checkCudaErrors(cudaMemcpyAsync(this->twf1 + this->copied_bytes, this->cu_twf + this->copied_bytes,
                    this->bytes_to_copy - this->copied_bytes, cudaMemcpyDeviceToHost, this->copy_stream));
            cudaStreamSynchronize(this->copy_stream);
        }
        this->copied_bytes = -1;
    }

    void finishWork() {
        this->startNewRound(-1, NULL);
        this->startNewRound(-1, NUL)
    }
};




#endif //ZSHDEMO_ASYNCCOMPUTINGUTILS_H
