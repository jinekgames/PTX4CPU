#pragma once

#include <string>
#include <unordered_map>

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

struct VirtualVar {
    VirtualVar() = default;
    VirtualVar(const VirtualVar&) = delete;
    VirtualVar(VirtualVar&& right) : ptxType(right.ptxType) {
        data.swap(right.data);
    }
    ~VirtualVar() = default;
    VirtualVar& operator = (const VirtualVar&) = delete;
    VirtualVar& operator = (VirtualVar&& right) {
        if (&right == this)
            return *this;
        data.swap(right.data);
        return *this;
    }

    std::string ptxType;
    std::unique_ptr<void*> data = nullptr;
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
