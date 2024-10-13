#include "virtual_var.h"


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
#ifdef COMPILE_SAFE_CHECKS
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

}  // namespace Types
}  // namespace PTX4CPU
