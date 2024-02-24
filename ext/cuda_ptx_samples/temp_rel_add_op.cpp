#include <stdint.h>
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\vector_types.h"

const uint3 tid = { 1, 1, 1 };















extern "C" void _Z9addKernelPiPKiS1_(
    const uint64_t _Z9addKernelPiPKiS1__param_0,
    const uint64_t _Z9addKernelPiPKiS1__param_1,
    const uint64_t _Z9addKernelPiPKiS1__param_2
)
{
    uint32_t r1, r2, r3, r4, r5;
    uint64_t rd1, rd2, rd3, rd4, rd5, rd6, rd7, rd8, rd9, rd10, rd11;


    rd1 = *((uint64_t*)(_Z9addKernelPiPKiS1__param_0));
    rd2 = *((uint64_t*)(_Z9addKernelPiPKiS1__param_1));
    rd3 = *((uint64_t*)(_Z9addKernelPiPKiS1__param_2));
    rd4 = (uint64_t)(rd1);
    rd5 = (uint64_t)(rd3);
    rd6 = (uint64_t)(rd2);
    r1 = (uint32_t)(tid.x);
    rd7 = (int32_t)(r1) * 4;
    rd8 = (int64_t)(rd6) + rd7;
    r2 = *((uint32_t*)(rd8));
    rd9 = (int64_t)(rd5) + rd7;
    r3 = *((uint32_t*)(rd9));
    r4 = (int32_t)(r3) + r2;
    rd10 = (int64_t)(rd4) + rd7;
    *((uint32_t*)(rd10)) = r4;
    return;

}