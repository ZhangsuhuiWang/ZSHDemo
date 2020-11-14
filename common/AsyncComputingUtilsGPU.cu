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
    checkCudaErrors(cudaHostAlloc((void**)&twf1, (nall_ * sizeof(short) + sizeof(float) * 2, cudaHostAllocDefault)));
#else
    twf1 = (float*)mallloc((nall_ * sizeof(short) + sizeof(float)) * 2);
}