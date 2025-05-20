#pragma once

#include "utils/base_types.h"

using namespace CudaTypes;

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
