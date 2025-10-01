#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

void createInput(int m, int n){
    printf("generating input matrix of size %d x %d\n", m, n);
    printf("input matrix generated and saved to input.txt\n");
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
}


void writeOutput(int *mat, int r, int c){
    FILE *fptr;
    fptr = fopen("output.txt", "w");
    fprintf(fptr, "%d\n", r);
    fprintf(fptr, "%d\n", c);
    for (int i=0;i<r*c;i++){
        fprintf(fptr,"%d\n",mat[i]);
    }
    printf("csr decoded output matrix saved to output.txt\n");
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

__global__ void prefixSum(int *d_rowCounts, int *nnz){

    
}

int main(){

    int m = 10;
    int n = 10;
    createInput(m,n);

    int r,c;
    int *h_mat, *h_rowCounts, *h_row, *h_col, *h_val;
    int *d_mat, *d_rowCounts, *d_row, *d_col, *d_val;

    h_mat = getInput(&r,&c);

    h_rowCounts = (int*)malloc(r*sizeof(int));

    displayMatrix(h_mat,r,c);

    cudaMalloc((void**)&d_mat, r*c*sizeof(int));
    cudaMalloc((void**)&d_rowCounts, r*sizeof(int));
    cudaMemcpy(d_mat, h_mat, r*c*sizeof(int), cudaMemcpyHostToDevice);

    
    int blockSize = 32;
    int gridSize = (r + blockSize - 1) / blockSize;
    countNonZerosPerRow<<<gridSize, blockSize>>>(d_mat, d_rowCounts, r, c);
    cudaDeviceSynchronize();
    cudaMemcpy(h_rowCounts, d_rowCounts, r*sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < r; i++) {
        printf("rrow %d has %d non-zero elements\n", i, h_rowCounts[i]);
    }

    //need to implement prefix sum to get row array
    //add encoding and decoding kernels for csr
    //verify correcttness of encoding and decoding
    

    free(h_mat);
    cudaFree(d_mat);
    cudaFree(d_rowCounts);
    return 0;



    // cudaMalloc((void**)&d_mat, r*c*sizeof(int));
    // cudaMalloc((void**)&d_row, (r+1)*sizeof(int));

    // cudaMemcpy(d_mat, h_mat, r*c*sizeof(int), cudaMemcpyHostToDevice);

    // int block = 32;

    // int grid = ((r*c)+block-1)/block;

    // encodeCSR<<<grid,block>>>(d_mat, d_row, d_col, d_val, r, c);
    // cudaDeviceSynchronize();

    // cudaMemcpy(h_row, d_row, (r+1)*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_col, d_col, nnz*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_val, d_val, nnz*sizeof(int), cudaMemcpyDeviceToHost);

    // displayCSRMatrix(h_row,h_col,h_val,nnz,r);

    // //free memory
    // free(h_mat);
    // free(h_row);
    // free(h_col);
    // free(h_val);
    // cudaFree(d_mat);
    // cudaFree(d_row);
    // cudaFree(d_col);
    // cudaFree(d_val);
    // return 0;

}