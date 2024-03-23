#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <result.h>
#include <parser_types.h>


namespace PTX2ASM {

/**
 * Stores PTX source file and provide iteraion by instructions of a file and functions it is consisting of
*/
class Parser {

public:

    using Data            = ParserInternal::Data;
    using DataIterator    = ParserInternal::DataIterator;
    using Function        = ParserInternal::Function;
    using PtxProperties   = ParserInternal::PtxProperties;
    using VarsTable       = ParserInternal::VarsTable;
    using VirtualVar      = ParserInternal::VirtualVar;
    using VirtualVarsList = ParserInternal::VirtualVarsList;
    using VarPtxType      = ParserInternal::VarPtxType;

    // Preprocessing code type (each instruction into one line)
    using PreprocessData = std::list<std::string>;

    Parser() = default;
    /**
     * @param source source code of a PTX
    */
    Parser(const std::string& source);
    Parser(const Parser&) = delete;
    Parser(Parser&& right)
        : m_DataIter   (std::move(right.m_DataIter))
        , m_State      (std::move(right.m_State))
        , m_PtxProps   (std::move(right.m_PtxProps))
        , m_VarsTable  (std::move(right.m_VarsTable))
        , m_FuncsList  (std::move(right.m_FuncsList)) {

        right.m_State      = State::NotLoaded;
    }
    ~Parser() = default;

    Parser& operator = (const Parser&) = delete;
    Parser& operator = (Parser&& right) {
        if (&right == this)
            return *this;

        m_DataIter  = std::move(right.m_DataIter);
        m_State     = std::move(right.m_State);
        m_PtxProps  = std::move(right.m_PtxProps);
        std::swap(m_VarsTable, right.m_VarsTable);
        m_FuncsList = std::move(right.m_FuncsList);

        right.m_State      = State::NotLoaded;

        return *this;
    }

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
    static void ClearCodeComments(std::string& code);

    /**
     * Process "\" operators
    */
    static void ProcessLineTransfer(std::string& code);

    /**
     * Convert string code into list of instructions
    */
    static PreprocessData ConvertCode(std::string& code);
    /**
     * Convert list of instructions into vector type
    */
    static Data ConvertCode(const PreprocessData& code);

    /**
     * Preprocessing code
     * Includes:
     * - PTX props parcing
     *  @todo implementation: encrease support by:
     * - C-style directive processing
     * - include files processing
    */
    void PreprocessCode(PreprocessData& code) const;

    /**
     * Creates a virtual tables cosisted of
     * global defined variables
     * and defined funtions with their own virtual tables
    */
    bool InitVTable();

    /**
     * Allocates the memory for the functions arguments
    */
    static void AllocateFunctionsMemory() {};

private:

    DataIterator m_DataIter;

    enum class State {
        // Source is not loaded
        NotLoaded,
        // Preprocessing stage passed
        Preprocessed,
        // Code is ready for processing
        Loaded,
    };

    mutable State m_State = State::NotLoaded;

    mutable PtxProperties m_PtxProps;

    // list of dirictives which are not trailed by {} of ;
    inline static const std::vector<std::string> m_FreeDirictives = {
        ".version",
        ".target",
        ".address_size",
        // debug dirictives
        // @todo imlementation: @@DWARF dirictive
        // "@@DWARF",
        // @todo imlementation: .loc dirictive
        // ".loc",
        // C-style preprocessor dirictives
        // @todo imlementation: C-style dirictives
        // "#",
    };

    // list of dirictives which are efining a fucntion
    inline static const std::vector<std::string> m_FuncDefDirictives = {
        ".entry",
        ".func",
        ".function",
        // @todo imlementation: .callprototype dirictive
        // ".callprototype",
        // @todo imlementation: .alias dirictive
        // ".alias",
    };

    VarsTable m_VarsTable;

    static std::pair<std::string, VarPtxType> ParsePtxVar(const std::string& entry);

    // A list of functions stated in the PTX
    std::vector<Function> m_FuncsList;

};

};  // namespace PTX2ASM
