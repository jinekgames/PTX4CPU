#include "cuda_runtime.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cstdint>

#define CHECK_ERROR(x) assert(x == cudaError_t::cudaSuccess);


template<int block_size>
__global__ void  mmul(const double* A, uint32_t A_m, const double* B, uint32_t B_m, double* C) {
    uint32_t id_x = threadIdx.x;
    uint32_t id_y = threadIdx.y;

    double C_XY_Element = 0;

    for (int i = 0; i < block_size; i++)
        C_XY_Element += A[A_m * id_y + i] * B[B_m * i + id_x];

    C[id_y * B_m + id_x] = C_XY_Element;
}

void launch_cuda_mmul(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C) {
    const std::size_t block_size = 8;

    double* gpuA, * gpuB, * gpuC;

    // MEMORY ALLOC
    CHECK_ERROR(cudaMalloc((void**)&gpuA, A_n * A_m * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&gpuB, B_n * B_m * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&gpuC, A_n * B_m * sizeof(double)));

    // MEMORY COPY H to D
    CHECK_ERROR(cudaMemcpy(gpuA, A, A_n * A_m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(gpuB, B, B_n * B_m * sizeof(double), cudaMemcpyHostToDevice));


    dim3 blocks(B_m, A_n);
    dim3 grid(1, 1);

    mmul<block_size> << <grid, blocks >> > (gpuA, A_m, gpuB, B_m, gpuC);
    cudaDeviceSynchronize();
    // MEMORY COPY D to H
    CHECK_ERROR(cudaMemcpy(C, gpuC, A_n * B_m * sizeof(double), cudaMemcpyDeviceToHost));

    // FREE
    CHECK_ERROR(cudaFree(gpuA));
    CHECK_ERROR(cudaFree(gpuB));
    CHECK_ERROR(cudaFree(gpuC));
}


int main() {
    std::vector<double> A(64, 2);
    std::vector<double> B(64, 1);
    std::vector<double> C(64);

    launch_cuda_mmul(A.data(), 8, 8, B.data(), 8, 8, C.data());

    for (auto i : C) {
        assert(i == 16);
    }
    return 0;
}
