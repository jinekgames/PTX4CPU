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

Result Translator::ExecuteFunc(const std::string& funcName) {

    if (m_Parser.GetState() != Parser::State::Ready) {
        PRINT_E("Parser is not ready for execution");
        return {"Can't execute the kernel"};
    }

    // @todo implementation: pass args and thrds count

    const uint32_t count = 1;

    std::vector<uint32_t> vars[3];
    vars[0].resize(count);
    vars[1].resize(count);
    vars[2].resize(count);

    for (uint32_t i = 0; i < count; ++i) {
        vars[0][i] = 0;
        vars[1][i] = i + 1;
        vars[2][i] = i + 1;
    }

    uint64_t pVarsConv[] = { reinterpret_cast<uint64_t>(vars[0].data()),
                             reinterpret_cast<uint64_t>(vars[1].data()),
                             reinterpret_cast<uint64_t>(vars[2].data()) };

    uint64_t ppVarsConv[] = { reinterpret_cast<uint64_t>(&pVarsConv[0]),
                              reinterpret_cast<uint64_t>(&pVarsConv[1]),
                              reinterpret_cast<uint64_t>(&pVarsConv[2]) };

    Types::PTXVarList args;
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                &ppVarsConv[0]
            ))
        )
    );
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                &ppVarsConv[1]
            ))
        )
    );
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                &ppVarsConv[2]
            ))
        )
    );

    uint3_32 thrdsCount = { count, 1, 1 };

    PRINT_I("Executing kernel \"%s\" in block [%lu,%lu,%lu]",
            funcName.c_str(), thrdsCount.x, thrdsCount.y, thrdsCount.z);

    auto execs = m_Parser.MakeThreadExecutors(funcName, args, thrdsCount);

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

    // @todo implementation: save result data

    return {};
}
