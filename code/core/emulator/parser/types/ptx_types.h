#pragma once

#include <cstdint>
#include <stdfloat>
#include <string>


// @todo implemetation: check support with gcc
#define HAS_FIXED_FLOAT_SUPPORT (__STDCPP_FLOAT16_T__ == 1)


namespace PTX4CPU {
namespace Types {

namespace {

enum class PTXTypeBase : uint16_t {
    B     = 1 << 0,
    S     = 1 << 1,
    U     = 1 << 2,
    F     = 1 << 3,
    FHalf = 1 << 4,
    Pred  = 1 << 5,
};

enum class PTXTypeBitSize : uint16_t {
    b8    = 1 << 0,
    b16   = 1 << 1,
    b32   = 1 << 2,
    b64   = 1 << 3,
    b128  = 1 << 4,
};

} // anonimous namespace

enum class PTXType : uint32_t {
    None = 0,

    // untyped bits 8-bit
    B8    = (uint32_t)PTXTypeBase::B     | ((uint32_t)PTXTypeBitSize::b8   << 16),
    // untyped bits 16-bit
    B16   = (uint32_t)PTXTypeBase::B     | ((uint32_t)PTXTypeBitSize::b16  << 16),
    // untyped bits 32-bit
    B32   = (uint32_t)PTXTypeBase::B     | ((uint32_t)PTXTypeBitSize::b32  << 16),
    // untyped bits 64-bit
    B64   = (uint32_t)PTXTypeBase::B     | ((uint32_t)PTXTypeBitSize::b64  << 16),
    // untyped bits 128-bit
    B128  = (uint32_t)PTXTypeBase::B     | ((uint32_t)PTXTypeBitSize::b128 << 16),

    // signed integer 8-bit
    S8    = (uint32_t)PTXTypeBase::S     | ((uint32_t)PTXTypeBitSize::b8   << 16),
    // signed integer 16-bit
    S16   = (uint32_t)PTXTypeBase::S     | ((uint32_t)PTXTypeBitSize::b16  << 16),
    // signed integer 32-bit
    S32   = (uint32_t)PTXTypeBase::S     | ((uint32_t)PTXTypeBitSize::b32  << 16),
    // signed integer 64-bit
    S64   = (uint32_t)PTXTypeBase::S     | ((uint32_t)PTXTypeBitSize::b64  << 16),

    // unsigned integer 8-bit
    U8    = (uint32_t)PTXTypeBase::U     | ((uint32_t)PTXTypeBitSize::b8   << 16),
    // unsigned integer 16-bit
    U16   = (uint32_t)PTXTypeBase::U     | ((uint32_t)PTXTypeBitSize::b16  << 16),
    // unsigned integer 32-bit
    U32   = (uint32_t)PTXTypeBase::U     | ((uint32_t)PTXTypeBitSize::b32  << 16),
    // unsigned integer 64-bit
    U64   = (uint32_t)PTXTypeBase::U     | ((uint32_t)PTXTypeBitSize::b64  << 16),

    // floating-point 16-bit
    F16   = (uint32_t)PTXTypeBase::F     | ((uint32_t)PTXTypeBitSize::b16  << 16),
    // floating-point 16-bit half precision
    F16X2 = (uint32_t)PTXTypeBase::FHalf | ((uint32_t)PTXTypeBitSize::b32  << 16),
    // floating-point 32-bit
    F32   = (uint32_t)PTXTypeBase::F     | ((uint32_t)PTXTypeBitSize::b32  << 16),
    // floating-point 64-bit
    F64   = (uint32_t)PTXTypeBase::F     | ((uint32_t)PTXTypeBitSize::b64  << 16),

    // predicate
    Pred  = (uint32_t)PTXTypeBase::Pred,
};

inline PTXType operator & (PTXType left, PTXType right) {
    return static_cast<PTXType>(static_cast<uint32_t>(left) & static_cast<uint32_t>(right));
}
inline PTXType operator | (PTXType left, PTXType right) {
    return static_cast<PTXType>(static_cast<uint32_t>(left) | static_cast<uint32_t>(right));
}
inline constexpr PTXType operator & (PTXType left, uint32_t right) {
    return static_cast<PTXType>(static_cast<uint32_t>(left) & right);
}
inline constexpr PTXType operator | (PTXType left, uint32_t right) {
    return static_cast<PTXType>(static_cast<uint32_t>(left) | right);
}
inline constexpr bool operator != (PTXType left, uint32_t right) {
    return (static_cast<uint32_t>(left) != right);
}

inline constexpr PTXType GetDoubleSizeType(PTXType type) {

    // Already the maximum type size
    if ((type & static_cast<uint32_t>(PTXTypeBitSize::b128)) != 0)
        return type;

    const uint32_t typeBaseMask = 0xFFFF;

    const uint32_t typeBase = static_cast<uint32_t>(type & typeBaseMask);
    const uint32_t sizeBase = static_cast<uint32_t>(type & ~typeBaseMask);

    return static_cast<PTXType>(typeBase | (sizeBase << 1));
}

inline constexpr PTXType GetSystemPtrType() {
#ifdef SYSTEM_ARCH_64
    return PTXType::U64;
#else
    return PTXType::U32;
#endif
}

PTXType     StrToPTXType(const std::string& typeStr);
std::string PTXTypeToStr(PTXType type);

// @todo imlementation: use boost for uint128 and fixed floats (if non supported)
template<PTXType ptxType>
using getVarType =
    // byte types
    std::conditional_t<ptxType == PTXType::B8,     int8_t,
    std::conditional_t<ptxType == PTXType::B16,    int16_t,
    std::conditional_t<ptxType == PTXType::B32,    int32_t,
    std::conditional_t<ptxType == PTXType::B64,    int64_t,
    // std::conditional_t<ptxType == PTXType::B128,   int128_t, // unsupported

    // signed integer types
    std::conditional_t<ptxType == PTXType::S8,     int8_t,
    std::conditional_t<ptxType == PTXType::S16,    int16_t,
    std::conditional_t<ptxType == PTXType::S32,    int32_t,
    std::conditional_t<ptxType == PTXType::S64,    int64_t,

    // unsigned integer types
    std::conditional_t<ptxType == PTXType::U8,     uint8_t,
    std::conditional_t<ptxType == PTXType::U16,    uint16_t,
    std::conditional_t<ptxType == PTXType::U32,    uint32_t,
    std::conditional_t<ptxType == PTXType::U64,    uint64_t,

    // floating point types
#if HAS_FIXED_FLOAT_SUPPORT
    std::conditional_t<ptxType == PTXType::F16,    std::float16_t,
    std::conditional_t<ptxType == PTXType::F16X2,  std::bfloat16_t,
    std::conditional_t<ptxType == PTXType::F32,    std::float32_t,
    std::conditional_t<ptxType == PTXType::F64,    std::float64_t,
#else
    std::conditional_t<ptxType == PTXType::F16,    float,
    std::conditional_t<ptxType == PTXType::F16X2,  float,
    std::conditional_t<ptxType == PTXType::F32,    std::conditional_t<sizeof(float) == 4, float, double>,
    std::conditional_t<ptxType == PTXType::F64,    double,
#endif

    // default value
    uint64_t>>>>>>>>>>>>>>>>;

#define PTX_Internal_TypedOp_Construct_First(_runtimeType, _constType, ...) \
    if (_runtimeType == _constType) {                                       \
        constexpr PTX4CPU::Types::PTXType _PtxType_ = _constType;           \
        __VA_ARGS__                                                         \
    }

#define PTX_Internal_TypedOp_Construct_Middle(_runtimeType, _constType, ...) \
    else if (_runtimeType == _constType) {                                   \
        constexpr PTX4CPU::Types::PTXType _PtxType_ = _constType;            \
        __VA_ARGS__                                                          \
    }

#define PTX_Internal_TypedOp_Construct_Default(_runtimeType, ...)                                 \
    else {                                                                                        \
        PRINT_E("Unknown type PTXType(%d). Casting to .s64", static_cast<int32_t>(_runtimeType)); \
        constexpr PTX4CPU::Types::PTXType _PtxType_ = PTX4CPU::Types::PTXType::S64;               \
        __VA_ARGS__                                                                               \
    }

// Runs the passed code with a compile-time PTXType.
// Use `_PtxType_` as a compile-time type value
// @param type run-time type
// @param variadic_argument code which needs a compile-time type
#define PTXTypedOp(type, ...)                                                                        \
    do {                                                                                             \
        PTX_Internal_TypedOp_Construct_First(type,  PTX4CPU::Types::PTXType::B8,    __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::B16,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::B32,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::B64,   __VA_ARGS__)     \
        /*PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::B128,  __VA_ARGS__)*/ \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::S8,    __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::S16,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::S32,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::S64,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::U8,    __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::U16,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::U32,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::U64,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::F16,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::F16X2, __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::F32,   __VA_ARGS__)     \
        PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::F64,   __VA_ARGS__)     \
        /*PTX_Internal_TypedOp_Construct_Middle(type, PTX4CPU::Types::PTXType::Pred,  __VA_ARGS__)*/ \
        PTX_Internal_TypedOp_Construct_Default(type, __VA_ARGS__)                                    \
    } while (0);

}  // namespace Types
}  // namespace PTX4CPU


#if HAS_FIXED_FLOAT_SUPPORT
namespace std {
inline std::string to_string(const std::float16_t& value) {
    return std::to_string(static_cast<float>(value));
}
inline std::string to_string(const std::bfloat16_t& value) {
    return std::to_string(static_cast<float>(value));
}
}  // namespace std
#endif  // #if HAS_FIXED_FLOAT_SUPPORT
