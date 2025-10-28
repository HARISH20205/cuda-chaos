#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    int N = 4;
    size_t size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);

    printf("Input Matrix A:\n");
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i + 1;
        printf("%.1f ", h_A[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    printf("\nInput Matrix B:\n");
    for (int i = 0; i < N * N; ++i) {
        h_B[i] = (i + 1) * 0.5f;
        printf("%.1f ", h_B[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    matrixMultiplyCPU(h_A, h_B, h_C_ref, N);

    printf("\nGPU Output Matrix C:\n");
    for (int i = 0; i < N * N; ++i) {
        printf("%.1f ", h_C[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    printf("\nCPU Reference Matrix C:\n");
    for (int i = 0; i < N * N; ++i) {
        printf("%.1f ", h_C_ref[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    int correct = 1;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3)
            correct = 0;
    }

    if (correct)
        printf("\nResults Match!\n");
    else
        printf("\nResults Mismatch!\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}