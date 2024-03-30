#include <algorithm>
#include <sstream>
#include <thread>

#include <logger.h>
#include <string_utils.h>
#include <translator.h>


namespace PTX2ASM {

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
    // Types::VarsTable args;
    Types::PTXVarList args;
    int3 thrdsCount = { 1, 1, 1 };

    PRINT_I("Executing a kernel \"%s\" in block [%llu,%llu,%llu]",
            funcName.c_str(), thrdsCount.x, thrdsCount.y, thrdsCount.z);

    auto execs = m_Parser.MakeThreadExecutors(funcName, args, thrdsCount);

    if (execs.empty()) {
        PRINT_E("Failed to create kernel executors");
        return {"Can't execute the kernel"};
    }

    std::list<std::thread> threads;

    for (auto& exec : execs) {
        threads.push_back(std::thread{[&] {
            exec.Run();
        }});
    }

    for (auto& thrd : threads) {
        thrd.join();
    }

    // @todo implementation: return result data

    return {};
}

};  // namespace PTX2ASM
