#include <algorithm>
#include <sstream>
#include <thread>

#include <helpers.h>
#include <logger/logger.h>
#include <string_utils.h>
#include <translator.h>


using namespace PTX4CPU;

// Constructors and Destructors

Translator::Translator() {

    // InvalidateTranslation();
}

Translator::Translator(const std::string& source)
    : m_Parser(source) {

    // InvalidateTranslation();
    // SetSource(source);
}


// Public methods

Result Translator::ExecuteFunc(const std::string& funcName, PtxInputData* pArgs,
                               const BaseTypes::uint3_32& gridSize) {

    if (m_Parser.GetState() != Parser::State::Ready) {
        PRINT_E("Parser is not ready for execution");
        return {"Can't execute the kernel"};
    }

    PRINT_I("Executing kernel \"%s\" in block [%lu,%lu,%lu]",
            funcName.c_str(), gridSize.x, gridSize.y, gridSize.z);

    auto execs = m_Parser.MakeThreadExecutors(funcName, pArgs->execArgs, gridSize);

    if (execs.empty()) {
        PRINT_E("Failed to create kernel executors");
        return {"Can't execute the kernel"};
    }

    std::list<std::thread> threads;

    Helpers::Timer overallTimer("Overall execution");

    for (auto& exec : execs) {
        auto thread = std::thread{[&] {
            Helpers::Timer threadTimer(std::vformat("Thread [{},{},{}]", std::make_format_args(
                exec.GetTID().x, exec.GetTID().y, exec.GetTID().z
            )));
            auto res = exec.Run();
            if(res)
            {
                PRINT_I("ThreadExecutor[%lu,%lu,%lu]: Execution finished",
                        exec.GetTID().x, exec.GetTID().y, exec.GetTID().z);
            } else {
                PRINT_E("ThreadExecutor[%lu,%lu,%lu]: Execution falied. Error: %s",
                        exec.GetTID().x, exec.GetTID().y, exec.GetTID().z, res.msg.c_str());
            }
        }};
        threads.push_back(std::move(thread));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return {};
}
