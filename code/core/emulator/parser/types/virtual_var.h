#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <logger/logger.h>
#include <types/ptx_types.h>
#include <validator.h>

namespace PTX4CPU {
namespace Types {

class PTXVar;

using PTXVarPtr  = std::shared_ptr<PTXVar>;
using PTXVarList = std::vector<PTXVarPtr>;
using IndexType  = int64_t;

// Virtual variable with the access-key
using ArgumentPair = std::pair<std::shared_ptr<PTXVar>, char>;

template<IndexType size>
concept VectorSize = size >= 1 && size <= 4;

template<IndexType size>
concept VectorIndex = size >= 0 && size <= 3;

class PTXVar {

public:

    using RawValuePtrType = std::shared_ptr<void>;

protected:

    explicit PTXVar(RawValuePtrType valuePtr);

public:

    PTXVar(const PTXVar&)              = delete;
    PTXVar& operator = (const PTXVar&) = delete;

    PTXVar(PTXVar&& right);
    PTXVar& operator = (PTXVar&& right);

    ~PTXVar() = default;

private:

    static void Move(PTXVar& left, PTXVar& right);

public:

    template<PTXType ptxType>
    getVarType<ptxType>& Get(IndexType idx = 0) {
        return GetByIdx<ptxType>(idx);
    }

    template<PTXType ptxType>
    const getVarType<ptxType>& Get(IndexType idx = 0) const {
        return GetByIdx<ptxType>(idx);
    }

    template<PTXType ptxType>
    getVarType<ptxType>& Get(char key) {
        return GetByKey<ptxType>(key);
    }

    template<PTXType ptxType>
    const getVarType<ptxType>& Get(char key) const {
        return GetByKey<ptxType>(key);
    }

    template<PTXType ptxType>
    static getVarType<ptxType>& Get(ArgumentPair &arg) {
        return arg.first->GetByKey<ptxType>(arg.second);
    }

    virtual PTXVarPtr       MakeCopy()            = 0;
    virtual const PTXVarPtr MakeCopy()      const = 0;
    virtual PTXVarPtr       MakeReference()       = 0;
    virtual const PTXVarPtr MakeReference() const = 0;

    virtual constexpr PTXType   GetPTXType()    const = 0;
    // Size of static array
    virtual constexpr IndexType GetVectorSize() const = 0;
    // Size of dynamic array
    virtual IndexType GetDynamicSize() const = 0;

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

    template<PTXType type, bool copyAsReference>
    static bool AssignValue(ArgumentPair& dst, const ArgumentPair& src) {
        return CopyValue<type, type, copyAsReference>(*src.first, *dst.first,
                                                      src.second, dst.second);
    }

    virtual bool AssignValue(const void* pValue, char key = 'x') = 0;

    static bool AssignValue(ArgumentPair& arg, const void* pValue)
    {
        return arg.first->AssignValue(pValue, arg.second);
    }

    std::string ToStr() const;

protected:

    RawValuePtrType pValue;

private:

    template<PTXType ptxType>
    getVarType<ptxType>& GetByIdx(IndexType idx = 0) const {
#ifdef OPT_COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize() && idx >= GetDynamicSize()) {
            PRINT_E("Invalid vector access index (should be 0..%d). Treat as 0", GetVectorSize());
            idx = 0;
        }
#endif
        return static_cast<getVarType<ptxType>*>(pValue.get())[idx];
    }

    template<PTXType ptxType>
    getVarType<ptxType>& GetByKey(char key) const {
        constexpr std::array<char, 4> appropriateKeys = { 'x', 'y', 'z', 'w' };
        auto idx = static_cast<IndexType>(std::find(appropriateKeys.begin(), appropriateKeys.end(), key) - appropriateKeys.begin());
#ifdef OPT_COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize() && idx >= GetDynamicSize()) {
            PRINT_E("Invalid vector access key (should be one of \"xyzw\"). Treat as x");
            idx = 0;
        }
#endif
        return GetByIdx<ptxType>(idx);
    }

#ifdef DEBUG_BUILD
    void SetupDebugPtrs() {
        _debug_values = pValue.get();
    }

    union DebugValues
    {
        const void*     raw;

        const int8_t*   int8_ptr;
        const uint8_t*  uint8_ptr;
        const int16_t*  int16_ptr;
        const uint16_t* uint16_ptr;
        const int32_t*  int32_ptr;
        const uint32_t* uint32_ptr;
        const int64_t*  int64_ptr;
        const uint64_t* uint64_ptr;
        const float*    float_ptr;
        const double*   double_ptr;

        const int8_t**   ptr_int8_ptr;
        const uint8_t**  ptr_uint8_ptr;
        const int16_t**  ptr_int16_ptr;
        const uint16_t** ptr_uint16_ptr;
        const int32_t**  ptr_int32_ptr;
        const uint32_t** ptr_uint32_ptr;
        const int64_t**  ptr_int64_ptr;
        const uint64_t** ptr_uint64_ptr;
        const float**    ptr_float_ptr;
        const double**   ptr_double_ptr;

        DebugValues(const void* prt = nullptr) : raw{prt} {}
        DebugValues& operator = (const void* ptr) { raw = ptr; return *this; }
    } _debug_values;
#endif
};

template<PTXType ptxType>
struct PTXVarMemDeleter {
    using RealType = getVarType<ptxType>;
    void operator () (void* ptr) { delete[] static_cast<RealType*>(ptr); }
};

struct PTXVarMemDeleterNull {
    void operator () (void* ptr) {}
};

template<PTXType ptxType, IndexType VectorSize = 1,
         class Deleter = PTXVarMemDeleter<ptxType>>
class PTXVarTyped : public PTXVar {

public:

    using RealType = getVarType<ptxType>;

    PTXVarTyped()
        : _deleter{Deleter{}}
        , PTXVar{std::move(RawValuePtrType{
            static_cast<void*>(new RealType[VectorSize]),
            std::bind(&Deleter::operator(), &_deleter, std::placeholders::_1)
        })} {}
    explicit PTXVarTyped(const RealType* pInitValue)
        : _deleter{Deleter{}}
        , PTXVar{std::move(RawValuePtrType{
            static_cast<void*>(new RealType[VectorSize]),
            std::bind(&Deleter::operator(), &_deleter, std::placeholders::_1)
        })} {
        CopyVector(&Get(), pInitValue, std::make_integer_sequence<IndexType, VectorSize>{});
    }
    explicit PTXVarTyped(IndexType dynamicArraySize)
        : _deleter{Deleter{}}
        , PTXVar{std::move(RawValuePtrType{
            static_cast<void*>(new RealType[dynamicArraySize]),
            std::bind(&Deleter::operator(), &_deleter, std::placeholders::_1)
        })} {
        dynamicSize = dynamicArraySize;
    }

    PTXVarTyped(const PTXVarTyped&)              = delete;
    PTXVarTyped& operator = (const PTXVarTyped&) = delete;

    PTXVarTyped(PTXVarTyped&& right)              = default;
    PTXVarTyped& operator = (PTXVarTyped&& right) = default;

    ~PTXVarTyped() = default;

    constexpr PTXType   GetPTXType()    const override { return ptxType; }
    constexpr IndexType GetVectorSize() const override { return VectorSize; }
    IndexType GetDynamicSize() const { return dynamicSize; }

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

    bool AssignValue(const void* pValue, char key = 'x') override {
#ifdef OPT_COMPILE_SAFE_CHECKS
        if (!pValue) {
            PRINT_E("Trying to assign a value with invalid pointer (nullptr)");
            return false;
        }
#endif
        Get(key) = *static_cast<const RealType*>(pValue);
        return true;
    }

private:

    // Special private constuctor, creating a reference to the given raw value
    PTXVarTyped(RawValuePtrType pInitValue)
        : PTXVar{std::move(pInitValue)} {}

    template<IndexType... size>
    static void CopyVector(RealType* dst, const RealType* src,
                           std::integer_sequence<IndexType, size...> int_seq) {
        ((dst[size] = src[size]), ...);
    }

    IndexType dynamicSize = 1;

    Deleter _deleter;
};

class VarsTable {

public:

    explicit VarsTable(const VarsTable* pParentTable = nullptr);
    VarsTable(const VarsTable&) = delete;
    VarsTable(VarsTable&& right);
    ~VarsTable() = default;

    VarsTable& operator = (const VarsTable&) = delete;
    VarsTable& operator = (VarsTable&& right);

private:

    static void Move(VarsTable& left, VarsTable& right);

public:

    void SwapVars(VarsTable& table);
    bool Contains(const std::string& name);

    template<PTXType ptxType, IndexType VectorSize = 1>
    void AppendVar(const std::string& name) {
#ifdef OPT_COMPILE_SAFE_CHECKS
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
#ifdef OPT_COMPILE_SAFE_CHECKS
        if (virtualVars.contains(name)) {
            PRINT_W("Variable \"%s\" already exists in the current scope. "
                    "It will be overriden", name.c_str());
        }
#endif
        auto& var = virtualVars[name];
        PTXVarPtr newPtr{new PTXVarTyped<ptxType, VectorSize>(initValue)};
        var.swap(newPtr);
    }

    void AppendVar(const std::string& name, PTXVarPtr&& pVar);

    PTXVar& GetVar(const std::string& name);
    const PTXVar& GetVar(const std::string& name) const;

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

    void DeleteVar(const std::string& name);

    void Clear();

    PTXVarPtr       FindVar(const std::string& name);
    const PTXVarPtr FindVar(const std::string& name) const;

    const VarsTable* GetParent() const;

    std::string ToStr() const;

private:

    std::map<std::string, PTXVarPtr> virtualVars;

    const VarsTable* parent = nullptr;

};

struct PtxVarDesc {
    std::vector<std::string> attributes;
    PTXType type = PTXType::None;
};

template<PTXType type>
PTXVarPtr CreateTempValueVarTyped(const std::string& value) {

    std::stringstream ss(value);
    getVarType<type> realValue;
    ss >> realValue;

    return PTXVarPtr{new PTXVarTyped<type>{&realValue}};
}

template<PTXType type>
PTXVarPtr CreateTempVarFromPointerTyped(void* ptr) {

    auto* realPointer = reinterpret_cast<getVarType<type>*>(ptr);

    Validator::CheckPointer(realPointer);

    return PTXVarPtr{
        new Types::PTXVarTyped<type, 1, Types::PTXVarMemDeleterNull>{
            realPointer}};
}

}  // namespace Types
}  // namespace PTX4CPU


namespace std {
inline std::string to_string(const PTX4CPU::Types::PTXVar& var) {
    return var.ToStr();
}
inline std::string to_string(const PTX4CPU::Types::VarsTable& table) {
    return table.ToStr();
}
}  // namespace std
