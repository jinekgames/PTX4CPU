#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <executor.h>
#include <result.h>
#include <parser_types.h>
#include <parser_data.h>


namespace PTX4CPU {

/**
 * Stores PTX source file and provide iteraion by instructions of a file and functions it is consisting of
*/
class Parser {

public:

    // Preprocessing code type (each instruction into one line)
    using PreprocessData = std::list<std::string>;

    enum class State {
        // Source is not loaded
        NotLoaded,
        // Preprocessing stage passed
        Preprocessed,
        // Code is ready for processing
        Ready,
    };

    Parser() = default;
    /**
     * @param source source code of a PTX
    */
    Parser(const std::string& source);
    Parser(const Parser&) = delete;
    Parser(Parser&& right)
        : m_DataIter        {std::move(right.m_DataIter)}
        , m_State           {std::move(right.m_State)}
        , m_PtxProps        {std::move(right.m_PtxProps)}
        , m_GlobalVarsTable {std::move(right.m_GlobalVarsTable)}
        , m_FuncsList       {std::move(right.m_FuncsList)} {

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
        std::swap(m_GlobalVarsTable, right.m_GlobalVarsTable);
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

    State GetState() { return m_State; }

    /**
     * Prepare executors for each thread, which could be runned asynchronously
    */
    std::vector<ThreadExecutor> MakeThreadExecutors(const std::string& funcName, Types::PTXVarList&& arguments,
                                                    int3 threadsCount) const;

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
    static Data::Type ConvertCode(const PreprocessData& code);

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
    bool InitVTable() const;

private:

    mutable Data::Iterator m_DataIter;

    mutable State m_State = State::NotLoaded;

    mutable Types::PtxProperties m_PtxProps;

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

    // Global file variables
    mutable Types::VarsTable m_GlobalVarsTable;

    static std::pair<std::string, Types::PtxVarDesc> ParsePtxVar(const std::string& entry);

    // A list of functions stated in the PTX
    mutable Types::FuncsList m_FuncsList;

};

};  // namespace PTX4CPU
