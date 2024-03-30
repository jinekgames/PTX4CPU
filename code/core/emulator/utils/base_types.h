#pragma once

#include <cstdint>


template<class T>
struct _v3 {
    using type = T;

    type x = 0;
    type y = 0;
    type z = 0;
};

using int3 = _v3<int64_t>;