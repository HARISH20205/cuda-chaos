#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void prewittFilter(const unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}};
    int Gy[3][3] = {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}};

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int sumX = 0;
        int sumY = 0;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int pixel = input[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int val = (int)sqrtf((float)(sumX * sumX + sumY * sumY));

        val = val > 255 ? 255 : val;
        val = val < 0 ? 0 : val;

        output[y * width + x] = (unsigned char)val;
    }
}

int main()
{
    const int width = 6;
    const int height = 6;

    unsigned char h_input[width * height] = {
        10, 10, 10, 10, 10, 10,
        10, 10, 50, 50, 10, 10,
        10, 10, 100, 100, 10, 10,
        10, 10, 100, 100, 10, 10,
        10, 10, 50, 50, 10, 10,
        10, 10, 10, 10, 10, 10};

    unsigned char h_output[width * height] = {0};

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    cudaMemcpy(d_input, h_input, width * height, cudaMemcpyHostToDevice);

    dim3 blockSize = {16, 16, 1};
    dim3 gridSize = {(width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y,
                     1};

    prewittFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, width * height, cudaMemcpyDeviceToHost);

    printf("Output after Prewitt filter:\n");
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%d\t", (int)h_output[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
