#pragma once

#include <translator_interface.h>


namespace PTX2ASM {

class Translator : public ITranslator {

public:

    void SetSource(const std::string& source);
    bool Translate() override;
    std::string GetResult() const override;

private:
    /**
     * Set invalid state for traslation
    */
    void InvalidateTranslation();

public:
    Translator();
    /**
     * @param source source code of a PTX
    */
    Translator(const std::string& source);
    Translator(const Translator&) = delete;
    Translator(Translator&&) = delete;
    ~Translator() = default;

    Translator operator = (const Translator&) = delete;
    Translator operator = (Translator&&) = delete;

private:
    std::string m_PtxIn;
    std::string m_AsmOut;
    bool m_IsPtxTranslated;

};

};  // namespace PTX2ASM
