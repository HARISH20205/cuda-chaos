#include <stdio.h>
#include <cuda_runtime.h>



__global__ void kernel(float *a,float *b,float *c,int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<n){
        a[i] = blockDim.x;
        b[i] = blockIdx.x;
        c[i] = threadIdx.x;
    }
}


int main(){

    float *h_a,*h_b,*h_c;
    float *d_a,*d_b,*d_c;

    int N = 64;
    int block = 32;
    int grid = (N+block-1)/block;
    h_a = (float*)malloc(N*sizeof(float));
    h_b = (float*)malloc(N*sizeof(float));
    h_c = (float*)malloc(N*sizeof(float));
    
    cudaMalloc((void**)&d_a,N*sizeof(float));
    cudaMalloc((void**)&d_b,N*sizeof(float));
    cudaMalloc((void**)&d_c,N*sizeof(float));

    kernel<<<grid,block>>>(d_a,d_b,d_c,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nIndex | blockDim.x | blockIdx.x | threadIdx.x\n");
    printf("------+------------+------------+-------------\n");
    for(int i = 0; i < N; i++) {
        printf("%5d | %10.0f | %10.0f | %11.0f\n", i, h_a[i], h_b[i], h_c[i]);
    }
    printf("\n");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);   
}

