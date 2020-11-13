#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <getopt.h>
#include "consts.h"
#include "utils.h"
#include "gpu_utils.h"

using namespace std;

ProgramConf parse_program_arguments(int argc, char** argv) {
    const static char *optstring = "x:y:z:t:c";
    char opt = 0;
    ProgramConf configure;
    configure.checkCorrectness = false;
    while(opt = getopt(argc, argv, optstring) != -1) {
        switch(opt) {
            case 'x':
                configure.NX = atoi(optarg) + 2 * R;
                break;
            case 'y' :
                configure.NY = atoi(optarg) + 2 * R;
                break;
            case 'z':
                configure.NZ = atoi(optarg) + 2 * R;
                break;
            case 't':
                configure.timesteps = atoi(optarg);
                break;
            case 'c':
                configure.checkCorrectness = true;
                break;
            case '?':
                printf("Unknown option: %c\n", opt);
                exit(1);
            default:
                exit(2);
        }
    }
    printf("--- Configuration Report ---\n");
    printf("NX=%d, NY=%d, NZ=%d.\n", configure.NX - 2 * R, configure.NY - 2 * R, configure.NZ - 2 * R);
    printf("Array NX=%d, NY=%d, NZ=%d.\n", configure.NX, configure.NY, configure.NZ);
    printf("Timesteps: %d.\n", configure.timesteps);
    printf("Radius: %d.\n", R);
    printf("--- [DONE] ---\n");
    return configure;
}



int main(int argc, char** argv) {
    ProgramConf conf = parse_program_arguments(argc, argv);
    const int arr_size = conf.NX * conf.NY * conf.NZ;
    printf("开始随机初始化input_data...\n");
    float *input_data = new float[arr_size];
    float *coeff = new float[R + 1];

    fill_array_with_random_number(input_data, arr_size);
    fill_array_with_random_number(coeff, R + 1);

    float *dev_src, *dev_dst;

    const int padding = (128 / sizeof(float)) - R;
    const size_t padded_array_size = arr_size + padding;

    HANDLE_ERROR(cudaMalloc((void**)&dev_src, padded_array_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_dst, padded_array_size * sizeof(float)));

    float *dev_src_padding, *dev_dst_padding;


    delete []input_data;
    delete []coeff;


    exit(0);
}
