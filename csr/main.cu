#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>



void createInput(int m, int n){
    printf("generating input matrix of size %d x %d\n", m, n);
    printf("input matrix generated\n");
    srand(time(NULL));

    int min = 0;
    int max = 9;

    FILE *fptr;

    fptr = fopen("input.txt", "w");

    fprintf(fptr, "%d\n", m);
    fprintf(fptr, "%d\n", n);



    for (int i=0;i<m*n;i++){
        int r = (rand() % (10-1+1))+1;
        if (r>8){
            int num = (rand()% (max-min+1))+min;
            fprintf(fptr, "%d\n", num);
        }
        else{
            fprintf(fptr, "%d\n", 0);
        }

    }
    fclose(fptr);

    printf("input matrix saved successsfully\n");
}


void writeOutput(int *mat, int r, int c){
    FILE *fptr;
    fptr = fopen("output.txt", "w");
    fprintf(fptr, "%d\n", r);
    fprintf(fptr, "%d\n", c);
    for (int i=0;i<r*c;i++){
        fprintf(fptr,"%d\n",mat[i]);
    }
    printf("csr decoded output matrix saved successfully\n");
    fclose(fptr);
}



int* getInput( int *r, int *c){
    FILE *fptr;
    fptr = fopen("input.txt", "r");
    if (fptr == NULL){
        printf("file not found!\n");
        exit(1);
    }
    fscanf(fptr, "%d", r);
    fscanf(fptr, "%d", c);
    int* mat = (int*)malloc((*r)*(*c)*sizeof(int));
    if (!mat) {
        printf("memory allocation failed!\n");
        exit(1);
    }
    for (int i=0;i<((*r)*(*c));i++){
        fscanf(fptr,"%d",&mat[i]);
    }

    fclose(fptr);
    return mat;
}

void displayMatrix(int *mat, int r, int c){
    for (int i=0;i<r;i++){
        printf("| ");
        for (int j=0;j<c;j++){
            printf("%d ", mat[i*c+j]);
        }
        printf("|\n");
    }
}

void checkCSR(int *mat, int *dmat,int r, int c){
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++){
            if (mat[i*c+j]!=dmat[i*c+j]){
                printf("csr encoding/decoding failed!\n");
                return;
            }
        }
    }
    printf("csr encoding/decoding successful!\n");
}

void displayCSRMatrix(int *row, int *col, int *val, int nnz, int r) {
    printf("\ncsr representation:\n");
    printf("row array (size %d):\n", r + 1);
    for (int i = 0; i < r + 1; i++) {
        printf("%d ", row[i]);
    }
    printf("\ncol array (size %d):\n", nnz);
    for (int i = 0; i < nnz; i++) {
        printf("%d ", col[i]);
    }
    printf("\nval array (size %d):\n", nnz);
    for (int i = 0; i < nnz; i++) {
        printf("%d ", val[i]);
    }
    printf("\n");
}




__global__ void countNonZerosPerRow(int *d_mat, int *d_rowCounts, int r, int c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < r) {
        int count = 0;
        for (int j = 0; j < c; j++) {
            if (d_mat[row * c + j] != 0) count++;
        }
        d_rowCounts[row] = count;
    }
}


__global__ void encode(int *d_mat, int *d_row, int *d_col, int *d_val, int r, int c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= r) return;

    int start = d_row[row];
    int idx = 0;

    for (int col = 0; col < c; col++) {
        int val = d_mat[row * c + col];
        if (val != 0) {
            d_col[start + idx] = col;   
            d_val[start + idx] = val;  
            idx++;
        }
    
    }
}

__global__ void decode(int *d_mat, int *d_row, int *d_col, int *d_val, int r, int c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= r) return;

    int start = d_row[row];
    int end = d_row[row + 1];

    for (int i = start; i < end; i++) {
        int col = d_col[i];
        int val = d_val[i];
        d_mat[row * c + col] = val;
    }
}



int main(int argc, char* argv[]){

    int m;
    int n;
    cudaError_t err;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    printf("using matrix size %d x %d\n", m, n);
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    createInput(m,n);
    
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_create_input = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_create_input += (end.tv_nsec - start.tv_nsec) / 1e3;
    printf("time taken to create input: %f µs\n", elapsed_create_input);


    int r,c;
    int *h_mat, *h_rowCounts, *h_row, *h_col, *h_val,*hd_mat;
    int *d_mat, *d_rowCounts, *d_row, *d_col, *d_val, *dd_mat;


    clock_gettime(CLOCK_MONOTONIC, &start);

    h_mat = getInput(&r,&c);
    
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_get_input = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_get_input = (elapsed_get_input + (end.tv_nsec - start.tv_nsec)) / 1e3;
    printf("time taken to read input: %f µs\n", elapsed_get_input);

    h_rowCounts = (int*)malloc(r*sizeof(int));

    // displayMatrix(h_mat,r,c);
    cudaEventRecord(c_start);

    cudaMalloc((void**)&d_mat, r*c*sizeof(int));
    cudaMalloc((void**)&d_rowCounts, r*sizeof(int));
    cudaMemcpy(d_mat, h_mat, r*c*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);

    float elapsed_copy_input = 0;
    cudaEventElapsedTime(&elapsed_copy_input, c_start, c_stop);
    printf("time taken to copy input matrix to device: %f µs\n", elapsed_copy_input);


    cudaEventRecord(c_start);

    int blockSize = 512;
    int gridSize = (r + blockSize - 1) / blockSize;
    countNonZerosPerRow<<<gridSize, blockSize>>>(d_mat, d_rowCounts, r, c);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);

    float elapsed_count_non_zeros = 0;
    cudaEventElapsedTime(&elapsed_count_non_zeros, c_start, c_stop);
    printf("time taken to count non-zeros per row: %f µs\n", elapsed_count_non_zeros);

    cudaMemcpy(h_rowCounts, d_rowCounts, r*sizeof(int), cudaMemcpyDeviceToHost);

    h_row = (int *)malloc((r+1)*sizeof(int));
    h_row[0] = 0;
    for (int i = 1; i < r + 1; i++) {
        h_row[i] = h_row[i - 1] + h_rowCounts[i - 1];
    }
    
    int nnz = h_row[r];

    cudaMalloc((void**)&d_col, nnz * sizeof(int));
    cudaMalloc((void**)&d_val, nnz * sizeof(int));

    h_col = (int *)malloc(nnz * sizeof(int));
    h_val = (int *)malloc(nnz * sizeof(int));

    int block = 32;
    int grid = (r + block - 1) / block;

    cudaMalloc((void**)&d_row, (r + 1) * sizeof(int));
    cudaMemcpy(d_row, h_row, (r + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(c_start);


    encode<<<grid, block>>>(d_mat, d_row, d_col, d_val, r, c);
    cudaDeviceSynchronize();

    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);
    float elapsed_encode = 0;
    cudaEventElapsedTime(&elapsed_encode, c_start, c_stop);
    printf("time taken to encode matrix to csr: %f µs\n", elapsed_encode);

    cudaMemcpy(h_col, d_col, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val, d_val, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // displayCSRMatrix(h_row, h_col, h_val, nnz, r);

    cudaMalloc((void**)&dd_mat, r * c * sizeof(int));
    cudaMemset(dd_mat, 0, r * c * sizeof(int));

    cudaEventRecord(c_start);

    decode<<<grid, block>>>(dd_mat, d_row, d_col, d_val, r, c);
    cudaDeviceSynchronize();

    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);
    float elapsed_decode = 0;
    cudaEventElapsedTime(&elapsed_decode, c_start, c_stop);
    printf("time taken to decode csr to matrix: %f µs\n", elapsed_decode);


    hd_mat = (int*)malloc(r * c * sizeof(int));
    cudaMemcpy(hd_mat, dd_mat, r * c * sizeof(int), cudaMemcpyDeviceToHost);

    // displayMatrix(hd_mat, r, c);

    checkCSR(h_mat, hd_mat, r, c);

    free(h_mat);
    free(h_row);
    free(h_col);
    free(h_val);
    cudaFree(d_mat);
    cudaFree(d_rowCounts);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    return 0;
}