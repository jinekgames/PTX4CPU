#include "ptx_function.h"

#include "utils/string_ext/string_iterator.h"


namespace PTX4CPU {
namespace Types {

Instruction::Instruction(const std::string& intructionStr) {

    if (intructionStr.empty()) {
        PRINT_E("Got empty intruction string");
        return;
    }

    const StringIteration::SmartIterator iter{intructionStr};

    // Parse predicate
    auto predicateStr =
        iter.ReadWord(true, StringIteration::WordDelimiter::Space);
    if (predicateStr.front() == Instruction::Predicate::PRED_PREFIX_SYMB) {
        predicate = Instruction::Predicate{predicateStr};
        iter.Shift(predicateStr.length());
    }

    name = iter.ReadWord();

    while (iter.IsValid()) {
        constexpr auto delimiter = StringIteration::AllSpaces |
                                   StringIteration::Punct;
        const auto arg = iter.ReadWord(false, delimiter);
        if (!arg.empty()) {
            args.push_back(arg);
        }
    }
}

Instruction::Predicate::Predicate(const std::string& predicateStr) {

    if (predicateStr.empty()) {
        PRINT_E("Got empty predicate string");
        return;
    }

    // Sample string:  @!%predName:

    const StringIteration::SmartIterator iter{predicateStr};

    if (iter.GetChar() == PRED_NEGATIVE_SYMB) {
        isNegative = true;
        ++iter;
    } else {
        isNegative = false;
    }

    // Read 'till ':'
    varName = iter.ReadWord(false, StringIteration::WordDelimiter::Punct);
}

std::string Instruction::GetStrType() const
{
    return name.substr(name.find_last_of('.'));
}

Types::PTXType Instruction::GetPtxType() const
{
    return Types::StrToPTXType(GetStrType());
}

void Function::InsertInstructions(Data::Iterator& iter) {
    while (!iter.IsBlockEnd()) {
        decltype(auto) instructionStr = iter.ReadInstruction();
        const Types::Instruction instruction{instructionStr};
        instructions.push_back(instruction);
        ++iter;
    }
}

}  // namespace Types
}  // namespace PTX4CPU
