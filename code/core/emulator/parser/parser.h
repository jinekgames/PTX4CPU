#pragma once

#include <list>
#include <string>
#include <vector>

#include <result.h>


namespace PTX2ASM {

/**
 * Stores PTX source file and provide iteraion by instructions of a file and functions it is consisting of
*/
class Parser {

// Interating code type (base work) (each instruction into one line)
typedef std::vector<std::string> Data;
// Preprocessing code type (each instruction into one line)
typedef std::list<std::string> PreprocessData;

public:

    Parser() = default;
    /**
     * @param source source code of a PTX
    */
    Parser(const std::string& source);
    Parser(const Parser&) = delete;
    Parser(Parser&&)      = delete;
    ~Parser() = default;

    Parser operator = (const Parser&) = delete;
    Parser operator = (Parser&&)      = delete;

public:

    /**
     * Load sourcefile and preprocess:
     * - comments clearing
     * - empty lines clearing
     * - code converting into translation copatible type
     * TODO: directives and includes processing
    */
    Result Load(const std::string& source);

private:

    /**
     * Clears comments in a source file
    */
    void ClearCodeComments(std::string& code) const;

    /**
     * Process "\" operators
    */
    void ProcessLineTransfer(std::string& code) const;

    /**
     * Convert string code into list of instaructions
    */
    PreprocessData ConvertCode(std::string& code) const;

    /**
     * Preprocessing code
    */
    Data PreprocessCode(PreprocessData& code) const;

private:

    Data m_Data;

    enum class State {
        // Source is not loaded
        NotLoaded,
        // Code is loaded, ready for processing
        Loaded,
        // Preprocessing stage passed
        Preprocessed,
        // Code is translated and ready to be gotten
        Translated
    };

    State m_State = State::NotLoaded;

    struct PtxProperties {
        std::pair<uint8_t, uint8_t> version = { 0, 0 };
        uint8_t target      = 0;
        uint8_t addressSize = 0;
    };

    PtxProperties m_PtxProps;

    // list of dirictives which are not trailed by {} of ;
    const std::vector<std::string> m_FreeDirictives = {
        ".version",
        ".target",
        ".address_size",
        // debug dirictives
        "@@DWARF",
        ".loc",
        // C-style preprocessor dirictives
        "#",
    };

};

};  // namespace PTX2ASM
