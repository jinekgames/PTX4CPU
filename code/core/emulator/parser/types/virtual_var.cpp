#include "virtual_var.h"

#include <utils/string_utils.h>


namespace PTX4CPU {
namespace Types {

PTXVar::PTXVar(RawValuePtrType valuePtr)
    : pValue{std::move(valuePtr)} {

#ifdef DEBUG_BUILD
    SetupDebugPtrs();
#endif
}

PTXVar::PTXVar(PTXVar&& right) {

    Move(*this, right);
}

PTXVar& PTXVar::operator = (PTXVar&& right) {

    Move(*this, right);
    return *this;
}

void PTXVar::Move(PTXVar& left, PTXVar& right) {

    if (&left == &right) {
        return;
    }

    left.pValue = std::move(right.pValue);

#ifdef DEBUG_BUILD
    left.SetupDebugPtrs();
    right.SetupDebugPtrs();
#endif
}

std::string PTXVar::ToStr() const {

    std::string ret;

    const auto size     = GetVectorSize();
    const bool isVector = (size != 1);
    const auto type     = GetPTXType();

    if (isVector) {
        ret += "{ ";
    }

    for (PTX4CPU::Types::IndexType idx = 0; idx < size; ++idx) {

        std::string valueStr;

        PTXTypedOp(type,
            const auto value = Get<_PtxType_>(idx);
            valueStr = std::to_string(value);
        )

        ret += valueStr;

        if (idx != size - 1) {
            ret += ", ";
        }
    }

    if (isVector) {
        ret += " }";
    }

    return ret;
}

VarsTable::VarsTable(const VarsTable* pParentTable)
    : parent{pParentTable} {}

VarsTable::VarsTable(VarsTable&& right) {

    Move(*this, right);
    right.parent = nullptr;
}

VarsTable& VarsTable::operator = (VarsTable&& right) {

    Move(*this, right);
    return *this;
}

void VarsTable::Move(VarsTable& left, VarsTable& right) {

    if (&left == &right)
        return;

    left.virtualVars = std::move(right.virtualVars);
    left.parent = right.parent;
    right.parent = nullptr;
}

void VarsTable::SwapVars(VarsTable& table) {
    virtualVars.swap(table.virtualVars);
}

bool VarsTable::Contains(const std::string& name) {
    return bool{FindVar(name)};
}

void VarsTable::AppendVar(const std::string& name, PTXVarPtr&& pVar) {
#ifdef OPT_COMPILE_SAFE_CHECKS
    if (virtualVars.contains(name)) {
        PRINT_W("Variable \"%s\" already exists in the current scope. "
                "It will be overriden", name.c_str());
    }
#endif
    virtualVars[name] = std::move(pVar);
}

PTXVar& VarsTable::GetVar(const std::string& name) {
    return *FindVar(name);
}

const PTXVar& VarsTable::GetVar(const std::string& name) const {
    return *FindVar(name);
}

void VarsTable::DeleteVar(const std::string& name) {
    virtualVars.erase(name);
}

void VarsTable::Clear() {
    virtualVars.clear();
}

PTXVarPtr VarsTable::FindVar(const std::string& name) {
    for (const auto* pTable = this; pTable; pTable = pTable->parent) {
        if (pTable->virtualVars.contains(name))
            return pTable->virtualVars.at(name);
    }
    return nullptr;
}

const PTXVarPtr VarsTable::FindVar(const std::string& name) const {
    for (const auto* pTable = this; pTable; pTable = parent) {
        if (pTable->virtualVars.contains(name))
            return pTable->virtualVars.at(name);
    }
    return nullptr;
}

const VarsTable* VarsTable::GetParent() const { return parent; }

std::string VarsTable::ToStr() const {

    std::string ret;

    ret += "{\n";

    for (const auto varIt : virtualVars) {
        const auto& name     = varIt.first;
        const auto  pValue   = varIt.second;
        const auto  valueStr = (pValue)
                               ? pValue->ToStr()
                               : "nullptr";
        ret += FormatString("\"{}\": {}\n", name, valueStr);
    }

    if (parent) {
        ret += parent->ToStr();
    }

    ret += "}";

    return ret;
}

}  // namespace Types
}  // namespace PTX4CPU
