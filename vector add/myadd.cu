#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000
#define BLOCK_SIZE 512


void vector_add_cpu(float *a,float *b,float *c, int n){
    for (int i=0;i<n;i++){
        c[i] = a[i] * b[i];
    }
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        c[i] = a[i]+b[i];
    }
}



void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;

    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));

    init_vector(h_a,N);
    init_vector(h_b,N);

    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    int num_blocks = (N+BLOCK_SIZE - 1) /BLOCK_SIZE;

    double start_time = get_time();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    double end_time = get_time();
    
    double cpu_avg_time = (end_time-start_time);
 
 
    start_time = get_time();
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    double gpu_avg_time = (end_time-start_time);

    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);


    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
