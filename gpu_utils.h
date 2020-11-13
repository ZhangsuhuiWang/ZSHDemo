//
// Created by 王张苏徽 on 2020/11/10.
//

#ifndef ZSHDEMO_GPU_UTILS_H
#define ZSHDEMO_GPU_UTILS_H

static void __checkCudaErrors(cudaError_t err,
                        const char *file,
                        int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define checkCudaErrors( err ) (__checkCudaErrors( err, __FILE__, __LINE__ ))

#endif //ZSHDEMO_GPU_UTILS_H
