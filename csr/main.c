//so this is my version of csr encoding and decoding in c on cpu
//TODO: need to translate to cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void encoderCPU(int *mat, int r, int c, int *row, int*col, int *val){
    int idx = 0;
    row[0]=0;
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++){
            int id = i*c+j;
            if (mat[id]!=0){
                col[idx]=j;
                val[idx]=mat[id];
                idx++;
            }
        }
        row[i+1]=idx;
    }
}

int* decoderCPU(int *row, int *col, int *val, int nnz,int r, int c) {

    int *mat = (int*)calloc(r * c, sizeof(int));
    if (!mat) {
        printf("memory allocation failed!\n");
        exit(1);
    }
    for (int i = 0; i < r; i++) {
        for (int j = row[i]; j < row[i+1]; j++) {
            int colIdx = col[j];
            mat[i * c + colIdx] = val[j];
        }
    }
    return mat;
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

int countNonZero(int *mat, int r, int c){

    int count = 0;
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++){
            if (mat[i*c+j]!=0){
                count++;
            }
        }
    }
    return count;
}

void displayCSRMatrix(int *row, int *col, int *val, int nnz, int r) {
    printf("\csr representation:\n");
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


int main() {
    
    int m = 5;
    int n = 5;

    createInput(m,n);

    int r, c;
    int *mat  = getInput(&r, &c);
    displayMatrix(mat,r,c);


    int nnz = countNonZero(mat,r,c);

    printf("number of non zero elements(nnz): %d",nnz);

    int *row = (int*)malloc((r+1)*sizeof(int));
    int *col = (int*)malloc(nnz*sizeof(int));
    int *val = (int*)malloc(nnz*sizeof(int));


    encoderCPU(mat,r,c,row,col,val);

    displayCSRMatrix(row,col,val,nnz,r);

    int *dMat = decoderCPU(row,col,val,nnz,r,c);

    displayMatrix(dMat,r,c);

    writeOutput(dMat,r,c);

    checkCSR(mat,dMat,r,c);

    free(mat);
    free(row);
    free(col);
    free(val);
    free(dMat);

    return 0;
}
