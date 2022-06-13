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
    __shared__ _DOUBLE_ As[MC][KC];
    __shared__ _DOUBLE_ Bs[KC][NC];
    // Load fraction of shared memory into register
    _DOUBLE_ sAs[TM];
    _DOUBLE_ sBs[TN];
    _DOUBLE_ Cs[TM][TN] = {0};

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int I = by*MC + ty; 
    int J = bx*NC + tx;

    int end = N/KC + (N%KC!=0);

#pragma unroll 
    for(int kk=0; kk<end; kk++){
#pragma unroll
        for(int i=0; i<TM; i++){
#pragma unroll
            for(int j=0; j<KC/BLOCKDIM_X; j++){
                int ii = I + i*BLOCKDIM_Y;
                int jj = kk*KC+tx + j*BLOCKDIM_X;
                As[ty + i*BLOCKDIM_Y][tx + j*BLOCKDIM_X] = ii<N && jj<N ? A[ii*N + jj] : 0;
            }
        }
#pragma unroll
        for(int i=0; i<KC/BLOCKDIM_Y; i++){
#pragma unroll
            for(int j=0; j<TN; j++){
                int ii = kk*KC+ty + i*BLOCKDIM_Y;
                int jj = J + j*BLOCKDIM_X;
                Bs[ty + i*BLOCKDIM_Y][tx + j*BLOCKDIM_X] = ii<N && jj<N ? B[ii*N + jj] : 0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k=0; k<KC; k++){
#pragma unroll
            for(int i=0; i<TM; i++){
                sAs[i] = As[ty + BLOCKDIM_Y*i][k];
            }
#pragma unroll
            for(int i=0; i<TN; i++){
                sBs[i] = Bs[k][tx + BLOCKDIM_X*i];
            }
#pragma unroll
            for(int i=0; i<TM; i++){
#pragma unroll
                for(int j=0; j<TN; j++){
                    Cs[i][j] += sAs[i]*sBs[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int i=0; i<TM; i++){
#pragma unroll
        for(int j=0; j<TN; j++){
            int ii = I + i*BLOCKDIM_Y;
            int jj = J + j*BLOCKDIM_X;
            if(ii<N && jj<N){
                C[ii*N + jj] = Cs[i][j];
            }
        }
    }
}

__global__ void shared_matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
    __shared__ _DOUBLE_ As[TW][TW];
    __shared__ _DOUBLE_ Bs[TW][TW];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y; 
    int bx = blockIdx.x;
    int I = by*TW + ty; 
    int J = bx*TW + tx;
    double Cij = 0;
    
    int A_idx;
    int B_idx;
    for (int kk=0; kk<N/TW; kk++){
        A_idx = I*N+kk*TW+tx;
        B_idx = (kk*TW+ty)*N + J;

        // padding
        if(A_idx<N*N){
            As[ty][tx] = A[A_idx];
        }else{
            As[ty][tx] = 0;
        }
        if(B_idx<N*N){
            Bs[ty][tx] = B[B_idx];
        }else{
            Bs[ty][tx] = 0;
        }

        __syncthreads();
        for (int k=0; k<TW; k++)
            Cij+= As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if(I<N&&J<N){
        C[I*N + J] = Cij;
    }
}

__global__ void original_matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
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
