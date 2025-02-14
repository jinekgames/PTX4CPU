#include <algorithm>
#include <sstream>
#include <thread>

#include <helpers.h>
#include <logger/logger.h>
#include <string_utils.h>
#include <emulator/emulator.h>


using namespace PTX4CPU;

// Constructors and Destructors

Emulator::Emulator() {

    // InvalidateTranslation();
}

Emulator::Emulator(const std::string& source)
    : m_Parser(source) {

    // InvalidateTranslation();
    // SetSource(source);
}


// Public methods

Result Emulator::ExecuteFunc(const std::string& funcName,
                             PtxExecArgs args,
                             const BaseTypes::uint3_32& gridSize) {

    if (m_Parser.GetState() != Parser::State::Ready) {
        PRINT_E("Parser is not ready for execution");
        return {"Can't execute the kernel"};
    }

    PRINT_I("Executing kernel \"%s\" in block [%lu,%lu,%lu]",
            funcName.c_str(), gridSize.x, gridSize.y, gridSize.z);

    auto execs = m_Parser.MakeThreadExecutors(funcName, args->execArgs, gridSize);

    if (execs.empty()) {
        PRINT_E("Failed to create kernel executors");
        return {"Can't execute the kernel"};
    }

#ifndef OPT_SYNCHRONIZED_EXECUTION
    std::list<std::thread> threads;
#endif  // #ifndef OPT_SYNCHRONIZED_EXECUTION

    Helpers::Timer overallTimer("Overall execution");

    bool success = true;

    for (auto& exec : execs) {
        auto thread = std::thread{[&] {
            Helpers::Timer threadTimer(
                FormatString("Thread [{},{},{}]",
                             exec.GetTID().x, exec.GetTID().y, exec.GetTID().z));
            auto res = exec.Run();
            if (res) {
                PRINT_I("ThreadExecutor[%lu,%lu,%lu]: Execution finished",
                        exec.GetTID().x, exec.GetTID().y, exec.GetTID().z);
            } else {
                PRINT_E("ThreadExecutor[%lu,%lu,%lu]: Execution falied. "
                        "Error: %s",
                        exec.GetTID().x, exec.GetTID().y, exec.GetTID().z,
                        res.msg.c_str());
                success = false;
            }
        }};
#ifndef OPT_SYNCHRONIZED_EXECUTION
        threads.push_back(std::move(thread));
#else  // OPT_SYNCHRONIZED_EXECUTION
        thread.join();
#endif  // #ifndef OPT_SYNCHRONIZED_EXECUTION
    }

#ifndef OPT_SYNCHRONIZED_EXECUTION
    for (auto& thread : threads) {
        thread.join();
    }
#endif  // #ifdef OPT_SYNCHRONIZED_EXECUTION

    // @todo implementation: destroy arguments descriptor data

    if (success) {
        return {};
    }
    return { "Kernel execution failed" };
}

Result Emulator::GetKernelDescriptor(const std::string& name,
                                     Types::Function** pDescriptor) const {

    if(!pDescriptor) {
        return { "Null Descriptor pointer passed" };
    }

    *pDescriptor = PTX4CPU_NULL_HANDLE;

    if(m_Parser.GetState() != Parser::State::Ready) {
        return { "Retiriving kernel descriptor from a not loaded Emulator" };
    }

    auto* parserDescriptor = m_Parser.GetKernelDescription(name);
    if(!parserDescriptor) {
        return { "Kernel descriptor not found" };;
    }

    *pDescriptor = parserDescriptor;
    return {};
}
