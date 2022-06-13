// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{

    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;

    if ((I < N) && (J < N))
    {
        _DOUBLE_ _c = 0;
        for (unsigned int k = 0; k < N; k++)
        {
            _DOUBLE_ a = A[I * N + k];
            _DOUBLE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}
