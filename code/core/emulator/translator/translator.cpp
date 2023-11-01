#include <translator.h>


namespace PTX2ASM {

// Constructors and Destructors

Translator::Translator() {
    InvalidateTranslation();
}

Translator::Translator(const std::string& source) {
    InvalidateTranslation();
    SetSource(source);
}


// Private realizations

void Translator::InvalidateTranslation() {
    m_AsmOut.clear();
    m_IsPtxTranslated = false;
}


// Public realizations


void Translator::SetSource(const std::string& source) {
    m_PtxIn = source;
}

bool Translator::Translate() {
    InvalidateTranslation();
    // @todo a lot of shit (from m_PtxIn to m_AsmOut), set false if not translated
    m_AsmOut = m_PtxIn;
    m_IsPtxTranslated = true;

    return m_IsPtxTranslated;
}

std::string Translator::GetResult() const {
    if (m_IsPtxTranslated) {
        return m_AsmOut;
    }
    return "";
}

};  // namespace PTX2ASM
