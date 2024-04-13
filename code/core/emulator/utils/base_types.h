#pragma once

#include <cstdint>


template<class T>
struct _v3 {
    using type = T;

    type x = 0;
    type y = 0;
    type z = 0;
};

using int3     = _v3<int64_t>;
using uint3    = _v3<uint64_t>;
using uint3_32 = _v3<uint32_t>;

template<class T>
struct _v4 {
    using type = T;

    _v4(const _v3<T>& vec)
        : x{vec.x}
        , y{vec.y}
        , z{vec.z} {}

    type x = 0;
    type y = 0;
    type z = 0;
    type w = 0;
};

using int4     = _v4<int64_t>;
using uint4    = _v4<uint64_t>;
using uint4_32 = _v4<uint32_t>;