#pragma once

#include <array>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <executor.h>
#include <parser_types.h>
#include <utils/api_types.h>
#include <utils/result.h>


namespace PTX4CPU {

/**
 * Stores PTX source file and provide iteraion by instructions of a file and
 * functions it is consisting of
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
    Parser(Parser&& right) noexcept
        : m_DataIter        {std::move(right.m_DataIter)}
        , m_State           {std::move(right.m_State)}
        , m_PtxProps        {std::move(right.m_PtxProps)}
        , m_GlobalVarsTable {std::move(right.m_GlobalVarsTable)}
        , m_FuncsList       {std::move(right.m_FuncsList)} {

        right.m_State      = State::NotLoaded;
    }
    ~Parser() = default;

    Parser& operator = (const Parser&) = delete;
    Parser& operator = (Parser&& right) noexcept {
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

    inline State GetState() const { return m_State; }

    // Returns description of the given kernel
    Types::Function* GetKernelDescription(const std::string& name) const;

    /**
     * Prepare executors for each thread, which could be runned asynchronously
    */
    std::vector<ThreadExecutor> MakeThreadExecutors(const std::string& funcName,
                                                    const Types::PTXVarList& arguments,
                                                    BaseTypes::uint3_32 threadsCount) const;

    /**
     * Parses a PTX var from the input string `entry`
    */
    static Types::Function::ArgWithName ParsePtxVar(const std::string& entry);

    struct ParsedPtxVectorName {
        char        key = 'x';
        std::string name;
    };

    /**
     * Extracts an access key (should be one of xyzw) and real variable name
    */
    static ParsedPtxVectorName ParseVectorName(const std::string& name);

    /// @brief Removes `[]` operator from name
    /// @return `true` if there was dereferencing
    /// @note There should be no spaces
    static bool ExtractDereference(std::string& argName);

    static bool IsKernelFunction(const Types::Function& function);

    static bool IsLabel(const std::string& instructionStr);

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
    static Data::RawData ConvertCode(const PreprocessData& code);

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
     * For each function instructions list is inserted.
    */
    bool InitVTable();

    /**
     * Finds an appropriate funtion according to the specified signature
     * Retuns an iterator of the functions'table, pointing to the function
     * descriptor or the end iterator, if no fuction found
    */
    Types::FuncsList::iterator FindFunction(const std::string& funcName,
                                            const Types::PTXVarList& arguments) const;

private:

    mutable Data::Iterator m_DataIter;

    mutable State m_State = State::NotLoaded;

    mutable Types::PtxProperties m_PtxProps;

    struct Dirictives {
        static constexpr auto VERSION      = ".version";
        static constexpr auto TARGET       = ".target";
        static constexpr auto ADDRESS_SIZE = ".address_size";
        static constexpr auto DWARF        = "@@DWARF";
        static constexpr auto LOC          = ".loc";
        static constexpr auto C_STYLE      = "#";
    };

    // list of dirictives which are not trailed by {} of ;
    static constexpr std::array<std::string_view, 3> m_FreeDirictives {
        Dirictives::VERSION,
        Dirictives::TARGET,
        Dirictives::ADDRESS_SIZE,
        // debug dirictives
        // @todo imlementation: @@DWARF dirictive
        // Dirictives::DWARF,
        // @todo imlementation: .loc dirictive
        // Dirictives::LOC,
        // C-style preprocessor dirictives
        // @todo imlementation: C-style dirictives
        // Dirictives::C_STYLE,
    };

    struct KernelAttributes {
        static constexpr auto ENTRY          = ".entry";
        static constexpr auto FUNC           = ".func";
        static constexpr auto FUNCTION       = ".function";
        static constexpr auto CALLPROTOTYPE  = ".callprototype";
        static constexpr auto ALIAS          = ".alias";
    };

    // list of dirictives which are efining a fucntion
    static constexpr std::array<std::string_view, 3> m_FuncDefDirictives {
        KernelAttributes::ENTRY,
        KernelAttributes::FUNC,
        KernelAttributes::FUNCTION,
        // @todo imlementation: .callprototype dirictive
        // KernelAttributes::CALLPROTOTYPE,
        // @todo imlementation: .alias dirictive
        // KernelAttributes::ALIAS,
    };

    static bool IsFuncDefDirictive(const std::string& dirictive);

    static constexpr char m_LabelFrontSymbol = '$';

    // Global file variables
    mutable Types::VarsTable m_GlobalVarsTable;

    // A list of functions stated in the PTX
    mutable Types::FuncsList m_FuncsList;

};

};  // namespace PTX4CPU
