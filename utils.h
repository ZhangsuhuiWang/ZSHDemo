//
// Created by 王张苏徽 on 2020/11/10.
//

#ifndef ZSHDEMO_UTILS_H
#define ZSHDEMO_UTILS_H

#include <stdlib.h>

struct ProgramConf {
    int timesteps;
    int NX, NY, NZ;
    bool checkCorrectness;
};


void fill_array_with_random_number(float* a, int N)
{
    srand(42);
    for (size_t i = 0; i < N; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
    }
}


#endif //ZSHDEMO_UTILS_H
