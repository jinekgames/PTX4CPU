#pragma once


#include <vector>
#include <utility>

#include <translator_interface.h>
#include <parser.h>

namespace PTX2ASM {

class Translator : public ITranslator {

public:

    // void SetSource(const std::string& source);
    void ExecuteFunc(const std::string& funcName, ...) override {};

private:
    // /**
    //  * Set invalid state for traslation
    // */
    // void InvalidateTranslation();

public:
    Translator();
    /**
     * @param source source code of a PTX
    */
    Translator(const std::string& source);
    Translator(const Translator&) = delete;
    Translator(Translator&&)      = delete;
    ~Translator() = default;

    Translator operator = (const Translator&) = delete;
    Translator operator = (Translator&&)      = delete;

private:

    Parser m_Parser;

};

};  // namespace PTX2ASM
