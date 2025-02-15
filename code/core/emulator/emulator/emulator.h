#pragma once


#include <vector>
#include <utility>

#include <emulator/emulator_interface.h>
#include <parser.h>

namespace PTX4CPU {

class Emulator : public IEmulator {

public:

    // Execute a kernel with the given name from the loaded PTX
    Result ExecuteFunc(const std::string& funcName, PtxExecArgs args,
                       const BaseTypes::uint3_32& gridSize) override;

    // Retrives the description of a kernel with the given name
    Result GetKernelDescriptor(const std::string& name,
                               Types::Function** pDescriptor) const override;

public:

    /**
     * @param source source code of a PTX
    */
    explicit Emulator(const std::string& source);

    Emulator();
    Emulator(const Emulator&)  = delete;
    Emulator(Emulator&& right) = default;
    ~Emulator() override       = default;

    Emulator& operator = (const Emulator&)  = delete;
    Emulator& operator = (Emulator&& right) = default;

private:

    Parser m_Parser;

};

};  // namespace PTX4CPU
