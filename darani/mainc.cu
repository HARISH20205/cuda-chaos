#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define WIDTH 8
#define HEIGHT 8
__global__ void prewitt_kernel(const unsigned char* in, unsigned char* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int gx_kernel[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
        int gy_kernel[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
        int gx_val = 0, gy_val = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int pixel_val = in[(y + i) * width + (x + j)];
                gx_val += pixel_val * gx_kernel[i + 1][j + 1];
                gy_val += pixel_val * gy_kernel[i + 1][j + 1];
            }
        }
        int magnitude = (int)sqrtf((float)(gx_val * gx_val + gy_val * gy_val));
        out[y * width + x] = (magnitude > 255) ? 255 : magnitude;
    }
}
void prewitt_edge_detection(const unsigned char* h_in_img, unsigned char* h_out_img, int width, int height) {
    unsigned char *d_in_img, *d_out_img;
    size_t img_size = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_in_img, img_size);
    cudaMalloc((void**)&d_out_img, img_size);
    cudaMemset(d_out_img, 0, img_size);
    cudaMemcpy(d_in_img, h_in_img, img_size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock = {16, 16};
    dim3 numBlocks = {(width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y};
    prewitt_kernel<<<numBlocks, threadsPerBlock>>>(d_in_img, d_out_img, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_img, d_out_img, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in_img);
    cudaFree(d_out_img);
}
int main() {
    unsigned char input[HEIGHT][WIDTH] = {
        {10, 10, 10, 10, 10, 10, 10, 10},
        {10, 50, 50, 50, 50, 50, 50, 10},
        {10, 50, 100, 100, 100, 100, 50, 10},
        {10, 50, 100, 150, 150, 100, 50, 10},
        {10, 50, 100, 150, 150, 100, 50, 10},
        {10, 50, 100, 100, 100, 100, 50, 10},
        {10, 50, 50, 50, 50, 50, 50, 10},
        {10, 10, 10, 10, 10, 10, 10, 10}
    };
    unsigned char* h_input = (unsigned char*)input;
    unsigned char* h_output = (unsigned char*)malloc(WIDTH * HEIGHT);
    printf("Input Matrix:\n");
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) printf("%3d ", input[y][x]);
        printf("\n");
    }
    prewitt_edge_detection(h_input, h_output, WIDTH, HEIGHT);
    printf("\nOutput (Edge Detection):\n");
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) printf("%3d ", h_output[y * WIDTH + x]);
        printf("\n");
    }
    free(h_output);

}