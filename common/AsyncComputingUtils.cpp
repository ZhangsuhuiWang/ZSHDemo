//
// Created by 王张苏徽 on 2020/11/11.
//

#include <math.h>
#include <string.h>
#include "AsyncComputingUtils.h"

int current_phase = 0;
struct timeval back_start_time;

void report_delta_time(int forward, int it, int cflag, int pflag) {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    float delta = current_time.tv_sec - back_start_time.tv_sec + (current_time.tv_usec - back_start_time.tv_usec) / 1e6f;
}

void openmp_memcpy(void* __restrict__ dst, const void* __restrict__ src, const long len) {
    #pragma omp parallel for num_threads(2)
    for(long i = 0; i < len; i++) {
        ((char*)dst)[i] = ((char*)src)[i];
    }
}

void* async_snapshot_worker_main(void *arg) {
    AsyncWavefieldSnapshotCoordinator *coordinator = (AsyncWavefieldSnapshotCoordinator*)arg;
    struct SnapshotTask task;
    while(true) {
        coordinator->fetchTask(&task);
        if(task.writer == NULL) {
            coordinator->fetchTask();
            pthread_exit(0);
        }
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        unsigned long psize = (unsigned  long)task.len / 2;
        signed char* pu3 = task.writer->getBuffer();
        openmp_memcpy(pu3, task.src, psize);
        task.writer->save(pu3, psize, task.it);
        pu3 = task.writer->getBuffer();
        openmp_memcpy(pu3, (char*)task.src + psize, psize);
        task.writer->save(pu3, psize, task.it);
        gettimeofday(&t2, NULL);
        long tu1 = t1.tv_sec * 1000000 + t1.tv_usec;
        long tu2 = t2.tv_sec * 1000000 + t2.tv_usec;
        coordinator->finishTask();
    }
    return NULL;
}

void* async_copy_worker_main(void *arg) {
    AsyncWavefieldCopyCoordinator *coordinator = (AsyncWavefieldCopyCoordinator*)arg;
    struct SnapshotTask task;
    while(true) {
        coordinator->fetchTask(&task);
        if(task.src == NULL && task.dst == NULL && task.len == 0) {
            coordinator->finishTask();
            pthread_exit(0);
        }
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        openmp_memcpy(task.dst, task.src, task.len);
        gettimeofday(&t2, NULL);
        long tu1 = t1.tv_sec * 1000000 + t1.tv_usec;
        long tu2 = t2.tv_sec * 1000000 + t2.tv_usec;
        coordinator->finishTask();
    }
    return NULL;
}

void *async_imaging_worker_main(void *arg) {
    AsyncImagingCoordinator *coordinator = (AsyncImagingCoordinator*)arg;
    struct Imaging task;
    while(true) {
        coordinator->fetchTask(&task);
        if(task.gpolicy == NULL) {
            coordinator->finishTask();
            pthread_exit(0);
        }
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        task.gpolicy->imaging(task.v, taks.bu, task.fu);
        gettimeofday(&t2, NULL);
        long tu1 = t1.tv_sec * 1000000 + t1.tv_usec;
        long tu2 = t2.tv_sec * 1000000 + t2.tv_usec;
    }
    return NULL;
}

OverlappedCompressAndSnapshotCoordinator *global_overlapped_compress_and_snapshot_coordinator = NULL;
OverlappedDtoHCopyCoordinator *global_dtoh_coordinator = NULL;
OverlappedImagingCoordinator *global_overlapped_imaging_coordinator = NULL;

void *start_async_memset(void *arg) {
    long batch_size = 128 * 1024 * 1024;
    MemsetTask *task = (MemsetTask*)arg;
    for(long i = 0; i < task->len; ){
        long remain_len = task->len - i;
        long this_len = (remain_len > batch_size) ? batch_size : remain_len;
        memset((char*)(task->p) + i, 0, this_len);
        if(i % (10 * batch_size) == 0) {

        }
        i += batch_size;
    }
    return NULL;
}
