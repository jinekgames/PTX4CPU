#include <algorithm>
#include <sstream>

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
        return {"Can't execute a kernel"};
    }

    PRINT_I("Executing a kernel \"%s\"", funcName.c_str());

    uint64_t par0   = 1;
    uint64_t par1[] = { 1 };
    uint64_t par2[] = { 1 };

    return {};
}

};  // namespace PTX2ASM
