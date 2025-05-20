#include <cuda_runtime.h>
#include <cassert>
#include <stdio.h>

__global__ void vadd_kernel(int arr_size, int *gpu_arr)
{
    int i = threadIdx.x;
    if (i < arr_size)
        gpu_arr[i] += i;
}

void vadd_launcher(int arr_size, int *arr, int blocksPerGrid, int threadsPerBlock)
{
    cudaError_t err = cudaSuccess;

    int *gpu_arr;
    err = cudaMalloc((void **)&gpu_arr, arr_size * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("gpuX memory allocation error. ");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpu_arr, arr, arr_size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Array memory relocation error. Host to device.\n%s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    vadd_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr_size, gpu_arr);
    cudaDeviceSynchronize();

    err = cudaMemcpy(arr, gpu_arr, arr_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Array memory relocation error. Device to host.\n%s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(gpu_arr);
    if (err != cudaSuccess)
    {
        printf("Array destruction error.\n%s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    int arr_size = 32;
    int *arr = new int[arr_size];
    for (int i = 0; i < arr_size; i++)
        arr[i] = 0;

    vadd_launcher(arr_size, arr, 1, 32);
    for (int i = 0; i < arr_size; i++)
        assert(arr[i] == i);
}
