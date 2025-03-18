#pragma once

typedef void *cudaStream_t;
typedef uint8_t uint3;

enum cudaError_t
{
    cudaSuccess,
    cudaErrorMemoryAllocation,
};

enum class cudaMemcpyKind
{
    cudaMemcpyHostToDevice = 0,
    cudaMemcpyDeviceToHost = 1,
    cudaMemcpyDeviceToDevice = 2,
    cudaMemcpyHostToHost = 3
};

struct dim3
{
    uint32_t x, y, z;
    dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z) {}
};


extern "C"
{
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaError_t cudaFree(void *devPtr);
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    cudaError_t cudaMemcpyHostToDevice(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    cudaError_t cudaMemcpyDeviceToHost(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    cudaError_t cudaGetLastError();
    const char *cudaGetErrorString(cudaError_t error);
    cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                 void **args, size_t sharedMem, cudaStream_t stream);
    cudaError_t cudaDeviceSynchronize();
    void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char const *deviceFun,
                                const char *deviceName, int threadLimit, uint3 *tid, uint3 *bid,
                                int *sharedMem, int *stream, void **function, int **kernelParams);
    void __cudaRegisterFatBinaryEnd(void* fatCubinHandle);
}
