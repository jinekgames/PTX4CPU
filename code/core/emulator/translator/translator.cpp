#include <algorithm>
#include <sstream>
#include <thread>

#include <logger.h>
#include <string_utils.h>
#include <translator.h>


namespace PTX4CPU {

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

    Types::PTXVarList args;
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(0))
        )
    );
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(1))
        )
    );
    args.push_back(
        std::move(
            Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(2))
        )
    );

    int3 thrdsCount = { 1, 1, 1 };

    PRINT_I("Executing a kernel \"%s\" in block [%llu,%llu,%llu]",
            funcName.c_str(), thrdsCount.x, thrdsCount.y, thrdsCount.z);

    auto execs = m_Parser.MakeThreadExecutors(funcName, std::move(args), thrdsCount);

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

};  // namespace PTX4CPU
