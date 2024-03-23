#include <algorithm>
#include <sstream>

#include <logger.h>
#include <translator.h>
#include <utils/string_utils.h>


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

};  // namespace PTX2ASM
