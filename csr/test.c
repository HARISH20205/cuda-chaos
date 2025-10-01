#include <stdio.h>
#include <stdlib.h>

void Encoder(int* arr, int* row, int* col, int* val, int m,int n){
    int c = 0;
    for(int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            int id = i*n+j;
            if (arr[id]!=0){
                row[c]=i;
                col[c]=j;
                val[c] = arr[id];
                c++;
                // printf("%d %d %d %d\n",i,j,id,arr[id]);
            }
        }
    }
}

void displayMatrix(int* arr,int m,int n){
    for(int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            printf("%d ",arr[i*n+j]);
        }
        printf("\n");
    }
}
void displayArray(int* arr,int n){
    for(int i=0;i<n;i++){
        printf("%d ",arr[i]);
    }
    printf("\n");
}


int main(){
    int arr[4][5] = {
    {10, 20,  0,  0,  0},
    { 0, 30,  0, 40,  0}, 
    { 0,  0, 50, 60, 70},
    { 0,  0,  0,  0, 80}
};
    int* ptr = (int*)arr;
    int m = 4;
    int n=5;
    int c = 0;
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            if (arr[i][j]!=0){
                c+=1;
            }
        }
    }

    int *row = (int *)malloc(c*sizeof(int));
    int *col = (int *)malloc(c*sizeof(int));
    int *val = (int *)malloc(c*sizeof(int));

    printf("Original Matrix:\n");
    displayMatrix(ptr, m,n);

    Encoder(ptr, row, col, val,m, n);

    printf("\nCSR Representation:\n");
    printf("| Row | Col | Val |\n");
    printf("|-----|-----|-----|\n");
    for (int i = 0; i < c; i++) {
        printf("| %3d | %3d | %3d |\n", row[i], col[i], val[i]);
    }




}

