#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

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
     * Convert string code into list of instructions
    */
    PreprocessData ConvertCode(std::string& code) const;
    /**
     * Convert list of instructions into vector type
    */
    Data ConvertCode(const PreprocessData& code) const;

    /**
     * Preprocessing code
     * Includes:
     * - PTX props parcing
     * @todo Tobe implemented:
     * - C-style directive processing
     * - include files processing
    */
    void PreprocessCode(PreprocessData& code) const;

    /**
     * Creates a virtual tables cosisted of
     * global defined variables
     * and defined funtions with their own virtual tables
    */
    void InitVTable();

    class VarsTable;

    static void AllocateMemory(VarsTable& vtable);

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

    mutable State m_State = State::NotLoaded;

    struct PtxProperties {
        std::pair<int8_t, int8_t> version = { 0, 0 };
        int32_t target      = 0;
        int32_t addressSize = 0;

        bool IsValid() {
            return (version.first || version.second) &&
                   version.first >= 0 && version.second >= 0 &&
                   target      > 0 &&
                   addressSize > 0;
        }
    };

    mutable PtxProperties m_PtxProps;

    // list of dirictives which are not trailed by {} of ;
    inline static const std::vector<std::string> m_FreeDirictives = {
        ".version",
        ".target",
        ".address_size",
        // debug dirictives
        // "@@DWARF", // @todo non-implemented
        // ".loc", // @todo non-implemented
        // C-style preprocessor dirictives
        // "#", // @todo non-implemented
    };

    // list of dirictives which are efining a fucntion
    inline static const std::vector<std::string> m_FuncDefDirictives = {
        ".entry",
        ".func",
        ".function",
        // ".callprototype", // @todo non-implemented
        // ".alias", // @todo non-implemented
    };

    struct VirtualVar {
        std::string ptxType;
        std::unique_ptr<void*> data = nullptr;
    };

    // PTX variable name to it's data
    typedef std::map<std::string, VirtualVar> VirtualVarsList;

    class VarsTable : public VirtualVarsList {
    private:
        const VarsTable* parent = nullptr;
    };

    VarsTable m_VarsTable;

    struct VarPtxType {
        std::vector<std::string> attributes;
        std::string type;
    };

    static std::tuple<std::string, VarPtxType> ParsePtxVar(const std::string& entry);

    struct Function
    {
        std::string name;
        // function attribute to it's optional value
        std::unordered_map<std::string, std::string> attributes;
        // argument name to it's type
        std::unordered_map<std::string, VarPtxType> arguments;
        // returning value name to it's type
        std::unordered_map<std::string, VarPtxType> returns;
        // Index of m_Data pointed to the first instruction of the function body
        Data::size_type start = -1;
        // Index of m_Data pointed to the first index after the last instruction of the function body
        Data::size_type end   = -1;
        VarsTable vtable;
    };

    std::vector<Function> m_FuncsTable;


};

};  // namespace PTX2ASM
