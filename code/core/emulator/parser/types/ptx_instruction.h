#pragma once

#include <list>
#include <optional>
#include <string>

class Instruction {

public:

    Instruction(const std::string& instructionString);

    Instruction() = delete;
    Instruction(const Instruction&) = delete;
    Instruction(Instruction&&)      = delete;
    ~Instruction() = default;

    Instruction& operator = (const Instruction&) = delete;
    Instruction& operator = (Instruction&&)      = delete;

    enum class Type {
        Instruction,
        VarDefinition,
        Label,
    };

private:

    class Predicate {

        // Checks the predicate condition if it is presented
        bool operator () ();

    private:

        // Represents whether the positive or negative result is target
        bool m_Positive = true;
        // Variable name storing the processed parameter
        std::string m_Var;
    };

    // Predicative execution param (optional)
    Predicate m_Pred;
    // Instruction name
    std::string m_Name;
    // Instruction execution params // @todo implementation: mb use PtxVar ref
    std::list<std::string> m_Params;

};
