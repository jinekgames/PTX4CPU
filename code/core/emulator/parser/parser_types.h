#pragma once

#include <algorithm>
#include <array>
#include <map>
#include <stdfloat>
#include <string>
#include <unordered_map>
#include <utility>

#include <logger/logger.h>
#include <parser_data.h>


namespace PTX4CPU {
namespace Types {

struct PtxProperties {
    std::pair<int8_t, int8_t> version = { 0, 0 };
    int32_t target      = 0;
    int32_t addressSize = 0;

    bool IsValid() {
        return (version.first || version.second) &&
                version.first >= 0 && version.second >= 0 &&
                target      > 0 &&
                addressSize > 0;
    }
};

enum class PTXTypeBase : uint16_t {
    B     = 1 << 0,
    S     = 1 << 1,
    U     = 1 << 2,
    F     = 1 << 3,
    FHalf = 1 << 4,
    Pred  = 1 << 5,
};

enum class PTXTypeBitSize : uint32_t {
    b8    = 1 << 16,
    b16   = 1 << 17,
    b32   = 1 << 18,
    b64   = 1 << 19,
    b128  = 1 << 20,
};

enum class PTXType : uint32_t {
    None = 0,

    // untyped bits 8-bit
    B8    = (uint32_t)PTXTypeBase::B | (uint32_t)PTXTypeBitSize::b8,
    // untyped bits 16-bit
    B16   = (uint32_t)PTXTypeBase::B | (uint32_t)PTXTypeBitSize::b16,
    // untyped bits 32-bit
    B32   = (uint32_t)PTXTypeBase::B | (uint32_t)PTXTypeBitSize::b32,
    // untyped bits 64-bit
    B64   = (uint32_t)PTXTypeBase::B | (uint32_t)PTXTypeBitSize::b64,
    // untyped bits 128-bit
    B128  = (uint32_t)PTXTypeBase::B | (uint32_t)PTXTypeBitSize::b128,

    // signed integer 8-bit
    S8    = (uint32_t)PTXTypeBase::S | (uint32_t)PTXTypeBitSize::b8,
    // signed integer 16-bit
    S16   = (uint32_t)PTXTypeBase::S | (uint32_t)PTXTypeBitSize::b16,
    // signed integer 32-bit
    S32   = (uint32_t)PTXTypeBase::S | (uint32_t)PTXTypeBitSize::b32,
    // signed integer 64-bit
    S64   = (uint32_t)PTXTypeBase::S | (uint32_t)PTXTypeBitSize::b64,

    // unsigned integer 8-bit
    U8    = (uint32_t)PTXTypeBase::U | (uint32_t)PTXTypeBitSize::b8,
    // unsigned integer 16-bit
    U16   = (uint32_t)PTXTypeBase::U | (uint32_t)PTXTypeBitSize::b16,
    // unsigned integer 32-bit
    U32   = (uint32_t)PTXTypeBase::U | (uint32_t)PTXTypeBitSize::b32,
    // unsigned integer 64-bit
    U64   = (uint32_t)PTXTypeBase::U | (uint32_t)PTXTypeBitSize::b64,

    // floating-point 16-bit
    F16   = (uint32_t)PTXTypeBase::F | (uint32_t)PTXTypeBitSize::b16,
    // floating-point 16-bit half precision
    F16X2 = (uint32_t)PTXTypeBase::FHalf | (uint32_t)PTXTypeBitSize::b32,
    // floating-point 32-bit
    F32   = (uint32_t)PTXTypeBase::F | (uint32_t)PTXTypeBitSize::b32,
    // floating-point 64-bit
    F64   = (uint32_t)PTXTypeBase::F | (uint32_t)PTXTypeBitSize::b64,

    // Predicate
    Pred = PTXTypeBase::Pred,
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

namespace {

inline static const std::unordered_map<std::string, PTXType> StrTable = {
    { ".b8",    PTXType::B8    },
    { ".b16",   PTXType::B16   },
    { ".b32",   PTXType::B32   },
    { ".b64",   PTXType::B64   },
    // { ".b128",  PTXType::B128  }, // unsupported

    { ".s8",    PTXType::S8    },
    { ".s16",   PTXType::S16   },
    { ".s32",   PTXType::S32   },
    { ".s64",   PTXType::S64   },

    { ".u8",    PTXType::U8    },
    { ".u16",   PTXType::U16   },
    { ".u32",   PTXType::U32   },
    { ".u64",   PTXType::U64   },

    { ".f16",   PTXType::F16   },
    { ".f16x2", PTXType::F16X2 },
    { ".f32",   PTXType::F32   },
    { ".f64",   PTXType::F64   },

    // { ".pred",  PTXType::Pred  }, // unsupported
};

}

inline PTXType GetFromStr(const std::string& typeStr) {
    if (StrTable.contains(typeStr)) {
        return StrTable.at(typeStr);
    }
    else {
        PRINT_E("Unknown type of variable: \"%s\". Treating as .s64", typeStr.c_str());
        return Types::PTXType::S64;
    }
}

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

    // // unsigned integer types
    std::conditional_t<ptxType == PTXType::U8,     uint8_t,
    std::conditional_t<ptxType == PTXType::U16,    uint16_t,
    std::conditional_t<ptxType == PTXType::U32,    uint32_t,
    std::conditional_t<ptxType == PTXType::U64,    uint64_t,

    // // floating point types
    std::conditional_t<ptxType == PTXType::F16,    float,
    std::conditional_t<ptxType == PTXType::F16X2,  float,
    std::conditional_t<ptxType == PTXType::F32,    std::conditional_t<sizeof(float) == 4, float, double>,
    std::conditional_t<ptxType == PTXType::F64,    double,
    // C++23 not supported in msvc yet
    // std::conditional_t<ptxType == PTXType::F16,    std::float16_t,
    // std::conditional_t<ptxType == PTXType::F16X2,  std::bfloat16_t,
    // std::conditional_t<ptxType == PTXType::F32,    std::float32_t,
    // std::conditional_t<ptxType == PTXType::F64,    std::float64_t,

    // default value
    uint64_t>>>>>>>>>>>>>>>>;

#define PTX_Internal_TypedOp_Construct_First(runtimeType, constType, ...) \
    if (runtimeType == constType) {                                       \
        const Types::PTXType _Runtime_Type_ = constType;                  \
        __VA_ARGS__                                                       \
    }

#define PTX_Internal_TypedOp_Construct_Following(runtimeType, constType, ...) \
    else if (runtimeType == constType) {                                      \
        const Types::PTXType _Runtime_Type_ = constType;                      \
        __VA_ARGS__                                                           \
    }

#define PTX_Internal_TypedOp_Construct_Default(runtimeType, ...)                                 \
    else {                                                                                       \
        PRINT_E("Unknown type PTXType(%d). Casting to .s64", static_cast<int32_t>(runtimeType)); \
        const Types::PTXType _Runtime_Type_ = Types::PTXType::S64;                               \
        __VA_ARGS__                                                                              \
    }

// use _Runtime_Type_ as template argument
#define PTXTypedOp(type, ...)                                                              \
    do {                                                                                   \
        PTX_Internal_TypedOp_Construct_First(type,     Types::PTXType::B8,    __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B16,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B32,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B64,   __VA_ARGS__) \
        /*PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B128,  __VA_ARGS__)*/ \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S8,    __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S16,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S32,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S64,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U8,    __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U16,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U32,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U64,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F16,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F16X2, __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F32,   __VA_ARGS__) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F64,   __VA_ARGS__) \
        /*PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::Pred,  __VA_ARGS__)*/ \
        PTX_Internal_TypedOp_Construct_Default(type, __VA_ARGS__)                          \
    } while (0);

class PTXVar;

using PTXVarPtr  = std::unique_ptr<PTXVar>;
using PTXVarList = std::vector<PTXVarPtr>;


using IndexType = int8_t;
template<IndexType size>
concept VectorSize = size >= 1 && size <= 4;
template<IndexType size>
concept VectorIndex = size >= 0 && size <= 3;

class PTXVar {

public:

    using RawValuePtrType = std::shared_ptr<void>;
    // using RawValuePtrType = std::unique_ptr<void, void(*)(void*)>;

protected:

    explicit PTXVar(RawValuePtrType valuePtr)
        : pValue{std::move(valuePtr)} {
#ifdef DEBUG_BUILD
        SetupDebugPtrs();
#endif
    }

public:

    PTXVar(const PTXVar&) = delete;
    PTXVar(PTXVar&& right) : pValue{std::move(right.pValue)} {}
    ~PTXVar() = default;

    PTXVar& operator = (const PTXVar&) = delete;
    PTXVar& operator = (PTXVar&& right) {
        if(this == &right)
            return *this;

        pValue = std::move(right.pValue);

        return *this;
    }

    template<PTXType ptxType>
    getVarType<ptxType>& Get(IndexType idx = 0) {
#ifdef COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize()) {
            PRINT_E("Invalid vector access index (should be 0..%d). Treat as 0", GetVectorSize());
            idx = 0;
        }
#endif
        return static_cast<getVarType<ptxType>*>(pValue.get())[idx];
    }

    template<PTXType ptxType>
    const getVarType<ptxType>& Get(IndexType idx = 0) const {
#ifdef COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize()) {
            PRINT_E("Invalid vector access index (should be 0..%d). Treat as 0", GetVectorSize());
            idx = 0;
        }
#endif
        return static_cast<getVarType<ptxType>*>(pValue.get())[idx];
    }

    template<PTXType ptxType>
    getVarType<ptxType>& Get(char key) {
        constexpr std::array<char, 4> appropriateKeys = { 'x', 'y', 'z', 'w' };
        auto idx = static_cast<IndexType>(std::find(appropriateKeys.begin(), appropriateKeys.end(), key) - appropriateKeys.begin());
#ifdef COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize()) {
            PRINT_E("Invalid vector access key (should be one of \"xyzw\"). Treat as x");
            idx = 0;
        }
#endif
        return Get<ptxType>(idx);
    }

    template<PTXType ptxType>
    const getVarType<ptxType>& Get(char key) const {
        constexpr std::array<char, 4> appropriateKeys = "xyzw";
        auto idx = static_cast<IndexType>(std::find(appropriateKeys.begin(), appropriateKeys.end(), key) - appropriateKeys.begin());
#ifdef COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize()) {
            PRINT_E("Invalid vector access key (should be one of \"xyzw\"). Treat as x");
            idx = 0;
        }
#endif
        return Get<ptxType>(idx);
    }

    virtual PTXVarPtr       MakeCopy()            = 0;
    virtual const PTXVarPtr MakeCopy()      const = 0;
    virtual PTXVarPtr       MakeReference()       = 0;
    virtual const PTXVarPtr MakeReference() const = 0;

    virtual constexpr PTXType   GetPTXType()    const = 0;
    virtual constexpr IndexType GetVectorSize() const = 0;

    template<PTXType srcType, PTXType dstType, bool copyAsReference>
    static bool CopyValue(PTXVar& src, PTXVar& dst, char srcKey = 'x', char dstKey = 'x') {
        using dstValueType = getVarType<dstType>;

        if constexpr (copyAsReference) {
            if (srcKey != 'x' && dstKey != 'x') {
                PRINT_E("Referencing of the element of the vector type is not currently supported");
                return false;
            }
            dst.pValue = src.pValue;
#ifdef DEBUG_BUILD
            dst.SetupDebugPtrs();
#endif
        } else {
            dst.Get<dstType>(dstKey) = static_cast<dstValueType>(src.Get<srcType>(srcKey));
        }
        return true;
    }

    template<PTXType type, bool copyAsReference>
    bool AssignValue(PTXVar& src, char srcKey = 'x', char dstKey = 'x') {
        return CopyValue<type, type, copyAsReference>(src, *this, srcKey, dstKey);
    }

    virtual bool AssignValue(void* pValue, char key = 'x') = 0;

protected:

    RawValuePtrType pValue;

private:

#ifdef DEBUG_BUILD
    void SetupDebugPtrs() {
        auto rawPtr = pValue.get();
        _debug_int8_ptr   = static_cast<decltype(_debug_int8_ptr)>(rawPtr);
        _debug_uint8_ptr  = static_cast<decltype(_debug_uint8_ptr)>(rawPtr);
        _debug_int16_ptr  = static_cast<decltype(_debug_int16_ptr)>(rawPtr);
        _debug_uint16_ptr = static_cast<decltype(_debug_uint16_ptr)>(rawPtr);
        _debug_int32_ptr  = static_cast<decltype(_debug_int32_ptr)>(rawPtr);
        _debug_uint32_ptr = static_cast<decltype(_debug_uint32_ptr)>(rawPtr);
        _debug_int64_ptr  = static_cast<decltype(_debug_int64_ptr)>(rawPtr);
        _debug_uint64_ptr = static_cast<decltype(_debug_uint64_ptr)>(rawPtr);
        _debug_float_ptr  = static_cast<decltype(_debug_float_ptr)>(rawPtr);
        _debug_double_ptr = static_cast<decltype(_debug_double_ptr)>(rawPtr);
    }

    int8_t*   _debug_int8_ptr   = nullptr;
    uint8_t*  _debug_uint8_ptr  = nullptr;
    int16_t*  _debug_int16_ptr  = nullptr;
    uint16_t* _debug_uint16_ptr = nullptr;
    int32_t*  _debug_int32_ptr  = nullptr;
    uint32_t* _debug_uint32_ptr = nullptr;
    int64_t*  _debug_int64_ptr  = nullptr;
    uint64_t* _debug_uint64_ptr = nullptr;
    float*    _debug_float_ptr  = nullptr;
    double*   _debug_double_ptr = nullptr;
#endif
};

template<PTXType ptxType, IndexType VectorSize = 1>
class PTXVarTyped : public PTXVar {

public:

    using RealType = getVarType<ptxType>;

    PTXVarTyped()
        : PTXVar{std::move(RawValuePtrType{
             static_cast<void*>(new RealType[VectorSize]),
             VarMemDeleter
          })} {}
    PTXVarTyped(const RealType* initValue)
        : PTXVar{std::move(RawValuePtrType{
             static_cast<void*>(new RealType[VectorSize]),
             VarMemDeleter
          })} {
        CopyVector(&Get(), initValue, std::make_integer_sequence<IndexType, VectorSize>{});
    }
    PTXVarTyped(const PTXVarTyped&) = delete;
    PTXVarTyped(PTXVarTyped&& right) {}
    ~PTXVarTyped() = default;

    PTXVarTyped& operator = (const PTXVarTyped&) = delete;
    PTXVarTyped& operator = (PTXVarTyped&& right) {}

    constexpr PTXType   GetPTXType()    const override { return ptxType; }
    constexpr IndexType GetVectorSize() const override { return VectorSize; }

    RealType& Get(IndexType idx = 0) {
        return PTXVar::Get<ptxType>(idx);
    }

    const RealType& Get(IndexType idx = 0) const {
        return PTXVar::Get<ptxType>(idx);
    }

    RealType& Get(char key) {
        return PTXVar::Get<ptxType>(key);
    }

    const RealType& Get(char key) const {
        return PTXVar::Get<ptxType>(key);
    }

    PTXVarPtr MakeCopy() override {
        return PTXVarPtr{new PTXVarTyped{&Get()}};
    }

    const PTXVarPtr MakeCopy() const override {
        return PTXVarPtr{new PTXVarTyped{&Get()}};
    }

    PTXVarPtr MakeReference() override {
        return PTXVarPtr{new PTXVarTyped{pValue}};
    }

    const PTXVarPtr MakeReference() const override {
        return PTXVarPtr{new PTXVarTyped{pValue}};
    }

    bool AssignValue(void* pValue, char key = 'x') override {
#ifdef COMPILE_SAFE_CHECKS
        if (!pValue) {
            PRINT_E("Trying to assign a value with invalid pointer (nullptr)");
            return false;
        }
#endif
        Get(key) = *static_cast<RealType*>(pValue);
        return true;
    }

private:

    // Special private constuctor, creating a reference to the given raw value
    PTXVarTyped(RawValuePtrType pInitValue)
        : PTXVar{std::move(pInitValue)} {}

    static void VarMemDeleter(void* rawPtr) {
        delete[] static_cast<RealType*>(rawPtr);
    }


    template<IndexType... size>
    static void CopyVector(RealType* dst, const RealType* src,
                           std::integer_sequence<IndexType, size...> int_seq) {
        ((dst[size] = src[size]), ...);
    }
};

class VarsTable {

public:

    explicit VarsTable(const VarsTable* pParentTable = nullptr)
        : parent{pParentTable} {}
    VarsTable(const VarsTable&) = delete;
    VarsTable(VarsTable&& right)
        : virtualVars{std::move(right.virtualVars)} {

        parent = right.parent;
        right.parent = nullptr;
    }
    ~VarsTable() = default;

    VarsTable& operator = (const VarsTable&) = delete;
    VarsTable& operator = (VarsTable&& right) {
        if (&right == this)
            return *this;

        virtualVars = std::move(right.virtualVars);
        parent = right.parent;
        right.parent = nullptr;

        return *this;
    }

    void SwapVars(VarsTable& table) {
        virtualVars.swap(table.virtualVars);
    }

    bool Contains(const std::string& name) {
        return FindVar(name);
    }

    template<PTXType ptxType, IndexType VectorSize = 1>
    void AppendVar(const std::string& name) {
#ifdef COMPILE_SAFE_CHECKS
        if (virtualVars.contains(name)) {
            PRINT_W("Variable \"%s\" already exists in the current scope. "
                    "It will be overriden", name.c_str());
        }
#endif
        auto& var = virtualVars[name];
        PTXVarPtr newPtr{new PTXVarTyped<ptxType, VectorSize>()};
        var.swap(newPtr);
    }

    template<PTXType ptxType, IndexType VectorSize = 1>
    void AppendVar(const std::string& name, const getVarType<ptxType>* initValue) {
#ifdef COMPILE_SAFE_CHECKS
        if (virtualVars.contains(name)) {
            PRINT_W("Variable \"%s\" already exists in the current scope. "
                    "It will be overriden", name.c_str());
        }
#endif
        auto& var = virtualVars[name];
        PTXVarPtr newPtr{new PTXVarTyped<ptxType, VectorSize>(initValue)};
        var.swap(newPtr);
    }

    void AppendVar(const std::string& name, PTXVarPtr&& pVar) {
#ifdef COMPILE_SAFE_CHECKS
        if (virtualVars.contains(name)) {
            PRINT_W("Variable \"%s\" already exists in the current scope. "
                    "It will be overriden", name.c_str());
        }
#endif
        virtualVars[name] = std::move(pVar);
    }

    PTXVar& GetVar(const std::string& name) {
        return *FindVar(name);
    }

    const PTXVar& GetVar(const std::string& name) const {
        return *FindVar(name);
    }

    template<PTXType ptxType>
    auto& GetValue(const std::string& name, IndexType idx = 0) {
        return FindVar(name)->Get<ptxType>(idx);
    }

    template<PTXType ptxType>
    const auto& GetValue(const std::string& name, IndexType idx = 0) const {
        return FindVar(name)->Get<ptxType>(idx);
    }

    template<PTXType ptxType>
    auto& GetValue(const std::string& name, char key) {
        return FindVar(name)->Get<ptxType>(key);
    }

    template<PTXType ptxType>
    const auto& GetValue(const std::string& name, char key) const {
        return FindVar(name)->Get<ptxType>(key);
    }

    const VarsTable* GetParent() const { return parent; }

    void DeleteVar(const std::string& name) {
        virtualVars.erase(name);
    }

    void Clear() {
        virtualVars.clear();
    }

    PTXVar* FindVar(const std::string& name) {
        for (const auto* pTable = this; pTable; pTable = pTable->parent) {
            if (pTable->virtualVars.contains(name))
                return pTable->virtualVars.at(name).get();
        }
        return nullptr;
    }

    const PTXVar* FindVar(const std::string& name) const {
        for (const auto* pTable = this; pTable; pTable = parent) {
            if (pTable->virtualVars.contains(name))
                return pTable->virtualVars.at(name).get();
        }
        return nullptr;
    }

private:

    std::map<std::string, PTXVarPtr> virtualVars;

    const VarsTable* parent = nullptr;

};

struct PtxVarDesc {
    std::vector<std::string> attributes;
    PTXType type;
};

struct Function
{

    Function()                = default;
    Function(const Function&) = default;
    Function(Function&& right)
        : name       {std::move(right.name)}
        , attributes {std::move(right.attributes)}
        , arguments  {std::move(right.arguments)}
        , returns    {std::move(right.returns)}
        , start      {right.start}
        , end        {right.end}
    {
        right.start = Data::Iterator::Npos;
        right.end   = Data::Iterator::Npos;
    }
    ~Function()               = default;
    Function& operator = (const Function&) = delete;
    Function& operator = (Function&& right) {
        if (&right == this)
            return *this;

        name       = std::move(right.name);
        attributes = std::move(right.attributes);
        arguments  = std::move(right.arguments);
        returns    = std::move(right.returns);
        start      = right.start;
        end        = right.end;

        right.start = Data::Iterator::Npos;
        right.end   = Data::Iterator::Npos;

        return *this;
    }

    // A name of the function stated in the PTX file
    std::string name;
    // function attribute to it's optional value
    std::unordered_map<std::string, std::string> attributes;
    // argument name to it's type
    std::unordered_map<std::string, PtxVarDesc> arguments;
    // returning value name to it's type
    std::unordered_map<std::string, PtxVarDesc> returns;
    // Index of m_Data pointed to the first instruction of the function body
    Data::Iterator::Size start = Data::Iterator::Npos;
    // Index of m_Data pointed to the first index after the last instruction of the function body
    Data::Iterator::Size end   = Data::Iterator::Npos;

};

using FuncsList = std::vector<Types::Function>;

}  // namespace Types
}  // namespace PTX4CPU
