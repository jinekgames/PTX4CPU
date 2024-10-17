#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ptx_types.h"

#include <logger/logger.h>


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

    PTXVar(const PTXVar&) = delete;
    PTXVar(PTXVar&& right);
    ~PTXVar() = default;

    PTXVar& operator = (const PTXVar&) = delete;
    PTXVar& operator = (PTXVar&& right);

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

    virtual bool AssignValue(void* pValue, char key = 'x') = 0;

    static bool AssignValue(ArgumentPair& arg, void* pValue)
    {
        return arg.first->AssignValue(pValue, arg.second);
    }

protected:

    RawValuePtrType pValue;

private:

    template<PTXType ptxType>
    getVarType<ptxType>& GetByIdx(IndexType idx = 0) const {
#ifdef COMPILE_SAFE_CHECKS
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
#ifdef COMPILE_SAFE_CHECKS
        if (idx < 0 || idx >= GetVectorSize() && idx >= GetDynamicSize()) {
            PRINT_E("Invalid vector access key (should be one of \"xyzw\"). Treat as x");
            idx = 0;
        }
#endif
        return GetByIdx<ptxType>(idx);
    }

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
    explicit PTXVarTyped(const RealType* pInitValue)
        : PTXVar{std::move(RawValuePtrType{
            static_cast<void*>(new RealType[VectorSize]),
            VarMemDeleter
        })} {
        CopyVector(&Get(), pInitValue, std::make_integer_sequence<IndexType, VectorSize>{});
    }
    explicit PTXVarTyped(IndexType dynamicArraySize)
        : PTXVar{std::move(RawValuePtrType{
            static_cast<void*>(new RealType[dynamicArraySize]),
            VarMemDeleter
        })} {
        dynamicSize = dynamicArraySize;
    }
    PTXVarTyped(const PTXVarTyped&) = delete;
    PTXVarTyped(PTXVarTyped&& right) {}
    ~PTXVarTyped() = default;

    PTXVarTyped& operator = (const PTXVarTyped&) = delete;
    PTXVarTyped& operator = (PTXVarTyped&& right) {}

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

    IndexType dynamicSize = 1;
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

private:

    std::map<std::string, PTXVarPtr> virtualVars;

    const VarsTable* parent = nullptr;

};

struct PtxVarDesc {
    std::vector<std::string> attributes;
    PTXType type;
};

template<PTXType type>
PTXVarPtr CreateTempValueVarTyped(const std::string& value) {

    std::stringstream ss(value);
    getVarType<type> realValue;
    ss >> realValue;

    return PTXVarPtr{new PTXVarTyped<type>{&realValue}};
}

}  // namespace Types
}  // namespace PTX4CPU
