#include "ptx_function.h"

#include <parser.h>
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

std::optional<Instruction>
Instruction::Make(const std::string& intructionStr) {

    if (intructionStr.empty()) {
        PRINT_E("Got empty intruction string");
        return {};
    }

    // handle lable
    if (Parser::IsLabel(intructionStr)) {
        PRINT_E("Got lable instead of an intruction");
        return {};
    }

    return Instruction{intructionStr};
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

    // Read from '@' till ':'
    varName = iter.ReadWord(
        false, StringIteration::WordDelimiter::Punct);
    if (!varName.empty())
        varName.erase(varName.begin());
}

std::string Instruction::GetStrType() const
{
    return name.substr(name.find_last_of('.'));
}

Types::PTXType Instruction::GetPtxType() const
{
    return Types::StrToPTXType(GetStrType());
}

void Function::ProcessInstructions(Data::Iterator& iter) {

    for (; !iter.IsBlockEnd(); ++iter) {

        decltype(auto) instructionStr = iter.ReadInstruction();

        if (Parser::IsLabel(instructionStr)) {
            const StringIteration::SmartIterator instuctionIt{instructionStr};
            const auto labelName = instuctionIt.ReadWord();

            IndexType offset = instructions.size();

            labels[labelName] = offset;
            continue;
        }

        const auto instruction = Types::Instruction::Make(instructionStr);

        if (instruction.has_value()) {
            instructions.push_back(instruction.value());
        }
        else {
            PRINT_E("Failed to parse isntruction \"%s\"",
                    instructionStr.c_str());
        }
    }
}

}  // namespace Types
}  // namespace PTX4CPU
