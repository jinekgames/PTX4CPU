#pragma once


#include <vector>
#include <utility>

#include <emulator/emulator_interface.h>
#include <parser.h>

namespace PTX4CPU {

class Emulator : public IEmulator {

public:

    // Execute a kernel with the given name from the loaded PTX
    Result ExecuteFunc(const std::string& funcName, PtxInputData* pArgs,
                       const BaseTypes::uint3_32& gridSize) override;

public:

    Emulator();
    /**
     * @param source source code of a PTX
    */
    Emulator(const std::string& source);
    Emulator(const Emulator&)  = delete;
    Emulator(Emulator&& right) = default;
    ~Emulator() = default;

    Emulator& operator = (const Emulator&)  = delete;
    Emulator& operator = (Emulator&& right) = default;

private:

    Parser m_Parser;

};

};  // namespace PTX4CPU
