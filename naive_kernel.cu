//
// Created by 王张苏徽 on 2020/11/10.
//

#include "consts.h"
#include "utils.h"
#include "gpu_utils.h"

__constant__ float dev_coeff[R + 1];

__global__ void naive_kernel(const float *src, float *dst, const int NX, const int NY, const int NZ) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if(gx >= R && gx < NX - R && gy >= R && gy < NY - R && gz >= R && gz < NZ - R) {
        const int stride_Z = NX * NY;
        const int stride_Y = NX;
        const int goffset = gz * stride_Z + gy * stride_Y + gx;
        float value = dev_coeff[0] * src[goffset];

        //left
        #pragma unroll 4
        for(int ir = 1; ir <= R; ir++) {
            value += dev_coeff[ir] * src[goffset - ir];
        }

        //right
        #pragma unroll 4
        for(int ir = 1; ir <= R; ir++) {
            value += dev_coeff[ir] * src[goffset + ir];
        }

        //front
        #pragma unroll 4
        for(int ir = 1; ir <= R; ir++) {
            value += dev_coeff[ir] * src[goffset - r * stride_Y];
        }

        //back
        #pragma unroll 4
        for(int ir = 1; ir <= R; ir++) {
            value += dev_coeff[ir] * src[goffset + r * stride_Y];
        }

        //top
        #pragma unroll 4
        for(int ir = 1; ir <= R; ir++) {
            value += dev_coeff[ir] * src[goffset + r * stride_Z];
        }

        //down
        #pragma unroll 4
        for(int ir = 1; ir <= R; ir++) {
            value += dev_coeff[ir] * src[goffset - r * stride_Z];
        }

        dst[goffset] = value;
    }
}