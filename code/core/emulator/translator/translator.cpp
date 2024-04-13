#include <algorithm>
#include <sstream>
#include <thread>

#include <logger.h>
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

    uint64_t varsRaw[] = { 1, 2, 3 };

    Types::PTXVarList args;
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                reinterpret_cast<uint64_t>(&varsRaw[0])
            ))
        )
    );
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                reinterpret_cast<uint64_t>(&varsRaw[1])
            ))
        )
    );
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                reinterpret_cast<uint64_t>(&varsRaw[2])
            ))
        )
    );

    uint3_32 thrdsCount = { 1, 1, 1 };

    PRINT_I("Executing kernel \"%s\" in block [%llu,%llu,%llu]",
            funcName.c_str(), thrdsCount.x, thrdsCount.y, thrdsCount.z);

    auto execs = m_Parser.MakeThreadExecutors(funcName, args, thrdsCount);

    if (execs.empty()) {
        PRINT_E("Failed to create kernel executors");
        return {"Can't execute the kernel"};
    }

    std::list<std::thread> threads;

    for (auto& exec : execs) {
        threads.push_back(std::thread{[&] {
            auto res = exec.Run();
            if(res)
            {
                PRINT_I("ThreadExecutor[%llu,%llu,%llu]: Execution finished",
                        exec.GetTID().x, exec.GetTID().y, exec.GetTID().z);
            } else {
                PRINT_E("ThreadExecutor[%llu,%llu,%llu]: Execution falied. Error: %s",
                        exec.GetTID().x, exec.GetTID().y, exec.GetTID().z, res.msg.c_str());
            }
        }});
    }

    for (auto& thrd : threads) {
        thrd.join();
    }

    // @todo implementation: save result data

    return {};
}
