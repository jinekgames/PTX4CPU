#include <string.h>
#include <dlfcn.h>
#include <memory>

#include "runtime_def.h"
#include "logger/logger.h"
#include "event_handler.h"

namespace {
    auto Handler = std::make_unique<EventHandler>();
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    *devPtr = malloc(size);
<<<<<<< HEAD
    PRINT_V("src: %p", *devPtr);
    PRINT_V("size: %llu", size);
=======
>>>>>>> main
    if (*devPtr == nullptr)
    {
        return cudaError_t::cudaErrorMemoryAllocation;
    }
    PRINT_V("cudaMalloc Intercepted");
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaFree(void *devPtr)
{
    PRINT_V("cudaFree Intercepted");
    free(devPtr);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    PRINT_V("cudaMemcpy Intercepted");
    PRINT_V("src: %p", src);
    PRINT_V("dst: %p", dst);
    PRINT_V("count: %llu", count);
    memcpy(dst, src, count);
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaGetLastError()
{
    PRINT_V("cudaGetLastError Intercepted");
    return cudaError_t::cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t error)
{
    return "No error\n";
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream)
{
    uint32_t thread_count = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    Handler->SetArgs(args);
    Handler->SetGridSize({blockDim.x, blockDim.y, blockDim.z});
    Handler->EmuKernelLaunch();
    PRINT_V("cudaLaunchKernel Intercepted");
    return cudaError_t::cudaSuccess;
}

cudaError_t cudaDeviceSynchronize()
{
    PRINT_V("cudaDeviceSynchronize Intercepted");
    return cudaError_t::cudaSuccess;
}

void __cudaRegisterFatBinaryEnd(void* fatCubinHandle) {
    PRINT_V("__cudaRegisterFatBinaryEnd Intercepted");
    Handler->LoadPtx();
    decltype(auto) orig = reinterpret_cast<void(*)(void*)>(dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd"));
    if (!orig) PRINT_E("Error while loading original function.");
    orig(fatCubinHandle);
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char const *deviceFun,
                            const char *deviceName, int threadLimit, uint3 *tid, uint3 *bid,
                            int *sharedMem, int *stream, void **function, int **kernelParams)
{
    Handler->LoadPtx();
    Handler->InitEmulatorCore();
    Handler->SetKernelName(deviceName);
}
