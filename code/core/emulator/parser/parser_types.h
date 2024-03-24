#pragma once

#include <string>
#include <unordered_map>

#include <logger.h>
#include <parser_data.h>


namespace PTX2ASM {
namespace ParserInternal {

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

// @todo refactoring: move types to another file

enum PTXTypeQualifires {
    PTXTypeBit       = 1 << 0,
    PTXTypeSigned    = 1 << 1,
    PTXTypeUnsigned  = 1 << 2,
    PTXTypeFloat     = 1 << 3,
    PTXTypePred      = 1 << 4,

    PTXType8         = 1 << 8,
    PTXType16        = 1 << 9,
    PTXType16X2      = 1 << 10,
    PTXType32        = 1 << 11,
    PTXType64        = 1 << 12,
    PTXType128       = 1 << 13,
};

enum class PTXType : int32_t {
    None = 0,

    // untyped bits 8-bit
    B8   = PTXTypeBit | PTXType8,
    // untyped bits 16-bit
    B16  = PTXTypeBit | PTXType16,
    // untyped bits 32-bit
    B32  = PTXTypeBit | PTXType32,
    // untyped bits 64-bit
    B64  = PTXTypeBit | PTXType64,
    // untyped bits 128-bit
    B128 = PTXTypeBit | PTXType128,

    // signed integer 8-bit
    S8  = PTXTypeSigned | PTXType8,
    // signed integer 16-bit
    S16 = PTXTypeSigned | PTXType16,
    // signed integer 32-bit
    S32 = PTXTypeSigned | PTXType32,
    // signed integer 64-bit
    S64 = PTXTypeSigned | PTXType64,

    // unsigned integer 8-bit
    U8  = PTXTypeUnsigned | PTXType8,
    // unsigned integer 16-bit
    U16 = PTXTypeUnsigned | PTXType16,
    // unsigned integer 32-bit
    U32 = PTXTypeUnsigned | PTXType32,
    // unsigned integer 64-bit
    U64 = PTXTypeUnsigned | PTXType64,

    // floating-point 16-bit
    F16   = PTXTypeFloat | PTXType16,
    // floating-point 16-bit half precision
    F16X2 = PTXTypeFloat | PTXType16X2,
    // floating-point 32-bit
    F32   = PTXTypeFloat | PTXType32,
    // floating-point 64-bit
    F64   = PTXTypeFloat | PTXType64,

    // Predicate
    Pred = PTXTypePred,
};

inline PTXType operator & (PTXType left, PTXType right) {
    return static_cast<PTXType>(static_cast<int32_t>(left) & static_cast<int32_t>(right));
}

inline int32_t operator & (PTXType left, PTXTypeQualifires right) {
    return static_cast<int32_t>(left) & static_cast<int32_t>(right);
}

inline static std::unordered_map<std::string, PTXType> PTXTypesStrTable = {
    { ".b8",    PTXType::B8    },
    { ".b16",   PTXType::B16   },
    { ".b32",   PTXType::B32   },
    { ".b64",   PTXType::B64   },
    { ".b128",  PTXType::B128  },

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

    { ".pred",  PTXType::Pred  },
};

struct PTXRawVarBase {

    PTXRawVarBase(PTXType type) : realType{type} {}

    virtual void* Get() = 0;

    PTXType realType;
};

struct PTXRawVar8 : public PTXRawVarBase {
    PTXRawVar8(PTXType type) : PTXRawVarBase{type} {}

    void* Get() override {
        if (realType & PTXTypeSigned)
            return static_cast<void*>(&_signed);
        else if (realType & PTXTypeUnsigned)
            return static_cast<void*>(&_unsigned);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_bit);

        PRINT_E("Unknown PTXType(%d), trait as .s8", static_cast<int>(realType));
        return static_cast<void*>(&_signed);
    }

    union {
        int8_t  _signed;
        uint8_t _unsigned;
        int8_t  _bit; // @todo implementation: is it correct to bit as int?
    };
};

struct PTXRawVar16 : public PTXRawVarBase {
    PTXRawVar16(PTXType type) : PTXRawVarBase{type} {}

    void* Get() override {
        if (realType & PTXTypeSigned)
            return static_cast<void*>(&_signed);
        else if (realType & PTXTypeUnsigned)
            return static_cast<void*>(&_unsigned);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_bit);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_float);

        PRINT_E("Unknown PTXType(%d), trait as .s16", static_cast<int>(realType));
        return static_cast<void*>(&_signed);
    }

    union {
        int16_t         _signed;
        uint16_t        _unsigned;
        int16_t         _bit;
        float           _float; // @todo implementation: is it a correct float representation? use std::float16_t with c++ 23 <stdfloat>
    };
};

struct PTXRawVar32 : public PTXRawVarBase {
    PTXRawVar32(PTXType type) : PTXRawVarBase{type} {}

    void* Get() override {
        if (realType & PTXTypeSigned)
            return static_cast<void*>(&_signed);
        else if (realType & PTXTypeUnsigned)
            return static_cast<void*>(&_unsigned);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_bit);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_float);

        PRINT_E("Unknown PTXType(%d), trait as .s32", static_cast<int>(realType));
        return static_cast<void*>(&_signed);
    }

    union {
        int32_t         _signed;
        uint32_t        _unsigned;
        int32_t         _bit;
        float           _float;
    };
};

struct PTXRawVar64 : public PTXRawVarBase {
    PTXRawVar64(PTXType type) : PTXRawVarBase{type} {}

    void* Get() override {
        if (realType & PTXTypeSigned)
            return static_cast<void*>(&_signed);
        else if (realType & PTXTypeUnsigned)
            return static_cast<void*>(&_unsigned);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_bit);
        else if (realType & PTXTypeBit)
            return static_cast<void*>(&_float);

        PRINT_E("Unknown PTXType(%d), trait as .s64", static_cast<int>(realType));
        return static_cast<void*>(&_signed);
    }

    union {
        int64_t         _signed;
        uint64_t        _unsigned;
        int64_t         _bit;
        double          _float;
    };
};

struct VirtualVar {
    VirtualVar() = default;
    VirtualVar(const VirtualVar&) = delete;
    VirtualVar(VirtualVar&& right) : ptxType(right.ptxType) {
        data.swap(right.data);

        right.ptxType = "";
        right.data.reset();
    }
    ~VirtualVar() = default;
    VirtualVar& operator = (const VirtualVar&) = delete;
    VirtualVar& operator = (VirtualVar&& right) {
        if (&right == this)
            return *this;

        data.swap(right.data);

        right.ptxType = "";
        right.data.reset();

        return *this;
    }

    void AllocateTyped(PTXType type) {
        if (type & PTXType8)
            data.reset(new PTXRawVar8(type));
        else if (type & PTXType16)
            data.reset(new PTXRawVar16(type));
        else if (type & PTXType32)
            data.reset(new PTXRawVar16(type));
        else if (type & PTXType64)
            data.reset(new PTXRawVar16(type));
        else
        {
            PRINT_E("Unknown PTXType(%d), trait as .s64", static_cast<int>(type));
            data.reset(new PTXRawVar64(PTXType::S64));
        }
    }

    std::string ptxType;
    std::unique_ptr<PTXRawVarBase> data = nullptr;
};

// PTX variable name to it's data
using VirtualVarsList = std::map<std::string, VirtualVar>;

class VarsTable : public VirtualVarsList {
public:
    VarsTable() = default;
    VarsTable(const VarsTable&) = delete;
    VarsTable(VarsTable&& right) {
        parent = right.parent;
        right.parent = nullptr;
    }
    ~VarsTable() = default;
    VarsTable& operator = (const VarsTable&) = delete;
    VarsTable& operator = (VarsTable&& right) {
        if (&right == this)
            return *this;

        parent = right.parent;
        right.parent = nullptr;

        return *this;
    }

    VarsTable* parent = nullptr;
};

struct VarPtxType {
    std::vector<std::string> attributes;
    std::string type;
};

class Function
{
public:
    Function()                = default;
    Function(const Function&) = delete;
    Function(Function&& right)
        : name       {std::move(right.name)}
        , attributes {std::move(right.attributes)}
        , arguments  {std::move(right.arguments)}
        , returns    {std::move(right.returns)}
        , start      {right.start}
        , end        {right.end}
        , vtable     {std::move(right.vtable)}
    {
        right.start = DataIterator::Npos;
        right.end   = DataIterator::Npos;
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
        vtable     = std::move(right.vtable);

        right.start = DataIterator::Npos;
        right.end   = DataIterator::Npos;

        return *this;
    }

    // A name of the function stated in the PTX file
    std::string name;
    // function attribute to it's optional value
    std::unordered_map<std::string, std::string> attributes;
    // argument name to it's type
    std::unordered_map<std::string, VarPtxType> arguments;
    // returning value name to it's type
    std::unordered_map<std::string, VarPtxType> returns;
    // Index of m_Data pointed to the first instruction of the function body
    DataIterator::Size start = DataIterator::Npos;
    // Index of m_Data pointed to the first index after the last instruction of the function body
    DataIterator::Size end   = DataIterator::Npos;
    // A table of variabled defined into the function
    VarsTable vtable;
};

}  // namespace ParserInternal
}  // namespace PTX2ASM
