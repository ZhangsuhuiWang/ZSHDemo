//
// Created by 王张苏徽 on 2020/11/11.
//

#include <math.h>
#include "../gpu_utils.h"
#include "AsyncComputingUtils.h"

#define USE_PINNED_MEMORY

float *twf1 = NULL;
float *twf2 = NULL;
float *cu_twf = NULL;

void init_temporary_wavefield_buffers(unsigned nall_) {
#ifdef USE_PINNED_MEMORY
    checkCudaErrors(cudaHostAlloc((void**)&twf1, (nall_ * sizeof(short) + sizeof(float)) * 2, cudaHostAllocDefault));
#else
    twf1 = (float*)mallloc((nall_ * sizeof(short) + sizeof(float)) * 2);
#endif
    memset(twf1, 0, (nall_ * sizeof(short) + sizeof(float)) * 2);
#ifdef USE_PINNED_MEMORY
    checkCudaErrors(cudaHostAlloc((void**)&twf2, (nall_ * sizeof(short) + sizeof(float)) * 2, cudaHostAllocDefault));
#else
    twf2 = (float*)malloc((nall_ * sizeof(short) + sizeof(float)) * 2);
#endif
    memset(twf2, 0, (nall_ * sizeof(short) + sizeof(float)) * 2);

    if(twf1 == NULL || twf2 == NULL) {
        printf("Fail to allocate the page-locked host pack buffer with %d elements.", (int)nall_);
    }
    checkCudaErrors(cudaMalloc(&cu_twf, (nall_ * sizeof(short) + sizeof(float)) * 2));
    if(cu_twf == NULL) {
        printf("Fail to allocate the temporary wavefield buffer on GPU with %d elements.", (int)nall_);
    }
}

void delete_temporary_wavefield_buffers() {
    if(twf1 != NULL) {
#ifdef USE_PINNED_MEMORY
        checkCudaErrors(cudaFreeHost(twf1));
#else
        free(twf1);
#endif
        twf1 = NULL;
    }
    if(twf2 == NULL) {
#ifdef USE_PINNED_MEMORY
        checkCudaErrors(cudaFreeHost(twf2));
#else
        free(twf2);
#endif
        twf2 = NULL;
    }
}

#define SHORT_MAX 32760
#define nscb_ 4

//number of threads pre block, must be power of 2
const unsigned int nTPB = 512;
//number of thread blocks for find max kernel
const unsigned int MAX_KERNEL_BLOCKS = 64;

#define MIN(a, b) (((a) > (b)) : (b) : (a))
#define MAX(a, b) (((a) > (b)) : (a) : (b))

//device global variables used to find the maximum value of an array
__device__ volatile float dev_blk_abs_max[MAX_KERNEL_BLOCKS];
//the result of block-level max is stored in dev_blk_abs_max
__global__ void block_level_abs_max_kernel(const float* data, const unsigned int dsize) {
    __shared__ float sdata[nTPB];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float myval = 0.0f;
    while(i < dsize) {
        myval = MAX(fabs(data[i]), myval);
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = myval;
    __syncthreads();
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] = MAX(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if(tid == 0) {
        dev_blk_abs_max[blockIdx.x] = sdata[0];
    }
}

float gpu_find_abs_max(const float* dev_data, const unsigned int len) {
    float blk_abs_max[MAX_KERNEL_BLOCKS];
    memset(blk_abs_max, 0, sizeof(float) * MAX_KERNEL_BLOCKS);
    checkCudaErrors(cudaMemcpyToSymbol(dev_blk_abs_max, &blk_abs_max[0], sizeof(float ) * MAX_KERNEL_BLOCKS));
    block_level_abs_max_kernel<<<MAX_KERNEL_BLOCKS, nTPB>>>(dev_data, len);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpyFromSymbol(&blk_abs_max, dev_blk_abs_max, sizeof(float) * MAX_KERNEL_BLOCKS));
    float max = blk_abs_max[0];
    for(unsigned i = 1; i < MAX_KERNEL_BLOCKS; i++) {
        if(blk_abs_max[i] > max) {
            max = blk_abs_max[i];
        }
    }
    return max;
}


__global__ void shortpack_compress(const float* dev_u3, char *dev_out, const unsigned arr_size, const float absmax) {
    unsigned idx = threadIdx.x + (long)blockDim.x * blockIdx.x;
    if(idx >= arr_size) {
        return;
    }
    float scale = absmax ? SHORT_MAX / absmax: 0.0;
    float unscale = absmax ? 1.0 / scale: 0.0;
    if(idx == 0) {
        float *head = (float*)dev_out;
        head[0] = unscale;
    }
    short *out_data = (short*)(dev_out + sizeof(float ));
    float x = dev_u3[idx] * scale;
    out_data[idx] = (short )x;
}

unsigned int gpu_pack(const unsigned int arr_size, const float *dev_u3, char *dev_pu3) {
    float abs = gpu_find_abs_max(dev_u3, arr_size);
    unsigned int num_block = (arr_size + nTPB - 1) / nTPB;
    shortpack_compress<<<num_block, nTPB>>>(dev_u3, dev_pu3, arr_size, absmax);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    float unscale = -1;
    checkCudaErrors(cudaMemcpy(&unscale, dev_pu3, sizeof(float), cudaMemcpyDeviceToHost));
    if(isnan(unscale)) {
        printf("unscale $f is not a valid number.", unscale);
    }
    return 0;
}