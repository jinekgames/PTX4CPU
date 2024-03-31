#pragma once

#include <map>
#include <stdfloat>
#include <string>
#include <unordered_map>

#include <logger.h>
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

enum class PTXType : int32_t {
    None = 0,

    // untyped bits 8-bit
    B8,//   = PTXTypeBit | PTXType8,
    // untyped bits 16-bit
    B16,//  = PTXTypeBit | PTXType16,
    // untyped bits 32-bit
    B32,//  = PTXTypeBit | PTXType32,
    // untyped bits 64-bit
    B64,//  = PTXTypeBit | PTXType64,
    // untyped bits 128-bit
    B128,// = PTXTypeBit | PTXType128,

    // signed integer 8-bit
    S8,//  = PTXTypeSigned | PTXType8,
    // signed integer 16-bit
    S16,// = PTXTypeSigned | PTXType16,
    // signed integer 32-bit
    S32,// = PTXTypeSigned | PTXType32,
    // signed integer 64-bit
    S64,// = PTXTypeSigned | PTXType64,

    // unsigned integer 8-bit
    U8,//  = PTXTypeUnsigned | PTXType8,
    // unsigned integer 16-bit
    U16,// = PTXTypeUnsigned | PTXType16,
    // unsigned integer 32-bit
    U32,// = PTXTypeUnsigned | PTXType32,
    // unsigned integer 64-bit
    U64,// = PTXTypeUnsigned | PTXType64,

    // floating-point 16-bit
    F16,//   = PTXTypeFloat | PTXType16,
    // floating-point 16-bit half precision
    F16X2,// = PTXTypeFloat | PTXType16X2,
    // floating-point 32-bit
    F32,//   = PTXTypeFloat | PTXType32,
    // floating-point 64-bit
    F64,//   = PTXTypeFloat | PTXType64,

    // Predicate
    Pred,// = PTXTypePred,

    Size,
};

inline PTXType operator & (PTXType left, PTXType right) {
    return static_cast<PTXType>(static_cast<int32_t>(left) & static_cast<int32_t>(right));
}

inline static const std::unordered_map<std::string, PTXType> PTXTypesStrTable = {
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

#define PTX_Internal_TypedOp_Construct_First(runtimeType, constType, op) \
    if (runtimeType == constType) {                                      \
        const Types::PTXType _Runtime_Type_ = constType;                 \
        op;                                                              \
    }

#define PTX_Internal_TypedOp_Construct_Following(runtimeType, constType, op) \
    else if (type == constType) {                                            \
        const Types::PTXType _Runtime_Type_ = constType;                     \
        op;                                                                  \
    }

#define PTX_Internal_TypedOp_Construct_Default(runtimeType, op)                                  \
    else {                                                                                       \
        PRINT_E("Unknown type PTXType(%d). Casting to .s64", static_cast<int32_t>(runtimeType)); \
        const Types::PTXType _Runtime_Type_ = Types::PTXType::S64;                               \
        op;                                                                                      \
    }

// use _Runtime_Type_ as template argument
#define PTXTypedOp(type, op)                                                      \
    do {                                                                          \
        PTX_Internal_TypedOp_Construct_First(type,     Types::PTXType::B8,    op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B16,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B32,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B64,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::B128,  op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S8,    op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S16,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S32,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::S64,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U8,    op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U16,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U32,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::U64,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F16,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F16X2, op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F32,   op) \
        PTX_Internal_TypedOp_Construct_Following(type, Types::PTXType::F64,   op) \
        PTX_Internal_TypedOp_Construct_Default(type, op)                          \
    } while (0)

    // if (type == Types::PTXType::B8) {                            \
    //     const Types::PTXType _Runtime_Type_ = Types::PTXType::B8; \
    //     op;
    // }
    // else if (type == PTXType::B16)         \
    //     op<PTXType::B16>(__VA_ARGS__);     \
    // else if (type == PTXType::B32)         \
    //     op<PTXType::B32>(__VA_ARGS__);     \
    // else if (type == PTXType::B64)         \
    //     op<PTXType::B64>(__VA_ARGS__);     \
    // else if (type == == PTXType:)          \
    //     op<== PTXType:B128>(__VA_ARGS__);  \
    // else if (type == PTXType::S8)          \
    //     op<PTXType::S8>(__VA_ARGS__);      \
    // else if (type == PTXType::S16)         \
    //     op<PTXType::S16>(__VA_ARGS__);     \
    // else if (type == PTXType::S32)         \
    //     op<PTXType::S32>(__VA_ARGS__);     \
    // else if (type == PTXType::S64)         \
    //     op<PTXType::S64>(__VA_ARGS__);     \
    // else if (type == PTXType::U8)          \
    //     op<PTXType::U8>(__VA_ARGS__);      \
    // else if (type == PTXType::U16)         \
    //     op<PTXType::U16>(__VA_ARGS__);     \
    // else if (type == PTXType::U32)         \
    //     op<PTXType::U32>(__VA_ARGS__);     \
    // else if (type == PTXType::U64)         \
    //     op<PTXType::U64>(__VA_ARGS__);     \
    // else if (type == PTXType::F16)         \
    //     op<PTXType::F16>(__VA_ARGS__);     \
    // else if (type == PTXType::F16X)        \
    //     op<PTXType::F16X>(__VA_ARGS__);    \
    // else if (type == PTXType::F32)         \
    //     op<PTXType::F32>(__VA_ARGS__);     \
    // else if (type == PTXType::F64)         \
    //     op<PTXType::F64>(__VA_ARGS__);

class PTXVar {

public:

    // using RawValuePtrType = std::shared_ptr<void>;
    using RawValuePtrType = std::unique_ptr<void, void(*)(void*)>;

protected:

    explicit PTXVar(RawValuePtrType valuePtr) : pValue{std::move(valuePtr)} {}

public:

    // PTXVar(const PTXVar& right) : pValue{right.pValue} {}
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
    getVarType<ptxType>& Get() {
        return *static_cast<getVarType<ptxType>*>(pValue.get());
    }

    template<PTXType ptxType>
    const getVarType<ptxType>& Get() const {
        return *static_cast<getVarType<ptxType>*>(pValue.get());
    }

    virtual constexpr PTXType GetPTXType() const = 0;

private:

    RawValuePtrType pValue;

};

template<PTXType ptxType>
class PTXVarTyped : public PTXVar {

public:

    using RealType = getVarType<ptxType>;

    PTXVarTyped()
        : PTXVar{std::move(RawValuePtrType{
             static_cast<void*>(new RealType()),
             VarMemDeleter
          })} {}
    PTXVarTyped(RealType initValue)
        : PTXVar{std::move(RawValuePtrType{
             static_cast<void*>(new RealType(initValue)),
             VarMemDeleter
          })} {}
    PTXVarTyped(const PTXVarTyped&) = delete;
    PTXVarTyped(PTXVarTyped&& right) {}
    ~PTXVarTyped() = default;

    PTXVarTyped& operator = (const PTXVarTyped&) = delete;
    PTXVarTyped& operator = (PTXVarTyped&& right) {}

    constexpr PTXType GetPTXType() const override { return ptxType; }

private:

    static void VarMemDeleter(void* rawPtr) {
        delete static_cast<RealType*>(rawPtr);
    }

};

using PTXVarPtr = std::unique_ptr<PTXVar>;

using PTXVarList = std::vector<PTXVarPtr>;

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

    template<PTXType ptxType>
    void AppendVar(const std::string& name) {
        auto& var = virtualVars[name];
        PTXVarPtr newPtr{new PTXVarTyped<ptxType>()};
        var.swap(newPtr);
    }

    template<PTXType ptxType>
    void AppendVar(const std::string& name, getVarType<ptxType> initValue) {
        auto& var = virtualVars[name];
        PTXVarPtr newPtr{new PTXVarTyped<ptxType>(initValue)};
        var.swap(newPtr);
    }

    PTXVar& GetVar(const std::string& name) {
        return *FindVar(name);
    }

    const PTXVar& GetVar(const std::string& name) const {
        return *FindVar(name);
    }

    template<PTXType ptxType>
    auto& GetValue(const std::string& name) {
        return FindVar(name)->Get<ptxType>();
    }

    template<PTXType ptxType>
    const auto& GetValue(const std::string& name) const {
        return FindVar(name)->Get<ptxType>();
    }

    const VarsTable* GetParent() const { return parent; }

    void DeleteVar(const std::string& name) {
        virtualVars.erase(name);
    }

    void Clear() {
        virtualVars.clear();
    }

private:

    PTXVar* FindVar(const std::string& name) {
        for (const auto* pTable = this; pTable; pTable = parent) {
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
