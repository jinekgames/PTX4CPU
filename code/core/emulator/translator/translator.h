#pragma once


#include <vector>
#include <utility>

#include <translator_interface.h>
#include <parser.h>

namespace PTX4CPU {

class Translator : public ITranslator {

public:

    // Execute a kernel with the given name from the loaded PTX
    Result ExecuteFunc(const std::string& funcName) override;

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
    Translator(Translator&& right)
        : m_Parser(std::move(right.m_Parser)) {

        right.m_Parser = {};
    }
    ~Translator() = default;

    Translator operator = (const Translator&) = delete;
    Translator operator = (Translator&& right) {
        m_Parser = std::move(right.m_Parser);
        right.m_Parser = {};
    }

private:

    Parser m_Parser;

};

};  // namespace PTX4CPU
