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


// Private realizations

// void Translator::InvalidateTranslation() {

//     m_CppOut.clear();
//     if (static_cast<int>(m_State) >= static_cast<int>(State::Translated)) {
//         if (m_PtxIn.empty())
//             m_State = State::NotLoaded;
//         else
//             m_State = State::Loaded;
//     }
// }


// Public realizations

// void Translator::SetSource(const std::string& source) {

//     if (source.empty())
//         return;

//     m_PtxIn = source;
//     m_State = State::Loaded;
// }

// bool Translator::Translate() {

//     InvalidateTranslation();

//     Preprocess();

//     if (m_State != State::Preprocessed)
//         return false;



//     // @todo a lot of shit (from m_PtxIn to m_AsmOut), set false if not translated
//     m_CppOut = m_PtxIn;
//     m_State = State::Translated;

//     return m_State == State::Translated;
// }

// std::string Translator::GetResult() const {

//     if (m_State == State::Translated) {
//         return m_CppOut;
//     }
//     return "";
// }

};  // namespace PTX2ASM
