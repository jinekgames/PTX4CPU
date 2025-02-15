#include <parser.h>

#include <logger/logger.h>
#include <string_utils.h>


using namespace PTX4CPU;

// Constructors and Destructors

Parser::Parser(const std::string& source) {

    auto res = Load(source);
    if (!res) {
        PRINT_E(res.msg.c_str());
    }
    PRINT_I("PTX file is loaded and ready for execution");
}

// Public realizations

Result Parser::Load(const std::string& source) {

    m_State = State::NotLoaded;

    auto rawCode = source;
    // @todo optimization: replace with list of chars
    // @todo optimization: replace string references with pointers
    //                     to empty memory then
    // @todo implementation: add logging on non-supported dirictives

    ProcessLineTransfer(rawCode);
    ClearCodeComments(rawCode);
    auto code = ConvertCode(rawCode);

    PreprocessCode(code);
    auto data = ConvertCode(code);
    m_DataIter = {std::move(data)};

    if (!m_PtxProps.IsValid())
        return {"The source PTX file is missing PTX properties"};

    PRINT_I("PTX props: ver %d.%d sm_%d address size %d",
            m_PtxProps.version.first, m_PtxProps.version.second,
            m_PtxProps.target, m_PtxProps.addressSize);

    if (!InitVTable())
        return {"Failed to init included functions table"};

    // @todo implementation: add global vars processing if existed

    m_State = State::Ready;

    return {};
}

Types::Function*
Parser::GetKernelDescription(const std::string& name) const {

    Types::Function* ret = nullptr;

    const auto it = std::find_if(m_FuncsList.begin(), m_FuncsList.end(),
        [&](const Types::FuncsList::value_type& func) {
            return IsKernelFunction(func) && func.name == name;
        });

    if (it != m_FuncsList.end()) {
        auto &description = *it;
        ret = &description;
    }

    return ret;
}

std::vector<ThreadExecutor>
Parser::MakeThreadExecutors(const std::string& funcName,
                            const Types::PTXVarList& arguments,
                            BaseTypes::uint3_32 threadsCount) const {

    // Find kernel
    auto funcIter = FindFunction(funcName, arguments);
    if (funcIter == m_FuncsList.end()) {
        PRINT_E("Function with such name (\"%s\") and corespondent arguments "
                "is not found in the PTX file", funcName.c_str());
        return {};
    }
    auto& func = *funcIter;

    // Convert arguments
    auto pAgumentsTable = std::make_shared<Types::VarsTable>(&m_GlobalVarsTable);
    Types::PTXVarList::size_type i = 0;
    for (auto& [name, var] : func.arguments) {
        pAgumentsTable->AppendVar(name, arguments[i]->MakeReference());
        ++i;
    }

    // Create executors
    std::vector<ThreadExecutor> ret;
    for (BaseTypes::uint3_32::type x = 0; x < threadsCount.x; ++x) {
        for (BaseTypes::uint3_32::type y = 0; y < threadsCount.y; ++y) {
            for (BaseTypes::uint3_32::type z = 0; z < threadsCount.z; ++z) {
                ret.push_back(std::move(
                    ThreadExecutor{&func, pAgumentsTable, {x, y, z}}
                ));
            }
        }
    }

    return ret;
}

std::pair<std::string, Types::PtxVarDesc> Parser::ParsePtxVar(const std::string& entry) {

    std::string name;
    Types::PtxVarDesc desc;
    // Sample string:
    // .reg .f64 dbl
    const StringIteration::SmartIterator iter{entry};

    desc.attributes.push_back(iter.ReadWord()); // contains memory location only

    auto typeStr = iter.ReadWord();
    desc.type = Types::StrToPTXType(typeStr);

    iter.Skip(StringIteration::CodeDelimiter);
    name = iter.ReadWord(false, StringIteration::AllSpaces);

    return {name, desc};
}

Parser::ParsedPtxVectorName Parser::ParseVectorName(const std::string& name) {

    ParsedPtxVectorName ret;
    StringIteration::SmartIterator iter{name};
    ret.name = iter.ReadWord(false, StringIteration::WordDelimiter::Dot);
    iter.Skip(StringIteration::WordDelimiter::Dot);
    if (iter.IsValid()) {
        ret.key = *iter.Current();
    } else {
        ret.key = 'x';
    }
    return ret;
}

bool Parser::ExtractDereference(std::string& argName) {

    StringIteration::SmartIterator iter{const_cast<const std::string&>(argName)};
    if (iter.IsBracket()) {
        argName = iter.ReadWord(false, StringIteration::Brackets);
        return true;
    }
    return false;
}

// Private realizations

void Parser::ClearCodeComments(std::string& code) {

    bool inSingLineComm = false;
    bool inMultLineComm = false;
    bool trueMultLineComm = false;
    size_t beginIdx = 0;
    for (size_t i = 0; i < code.length(); ++i) {
        if (inSingLineComm || inMultLineComm) {
            if (code[i] == LINE_ENDING.front() && inSingLineComm) {
                code.erase(beginIdx, i - beginIdx);
                i = beginIdx;
                inSingLineComm = false;
            } else if (inMultLineComm) {
                if (code[i] == LINE_ENDING.front()) {
                    trueMultLineComm = true;
                } else if (code[i] == '*' && code[i + 1] == '/') {
                    code.erase(beginIdx, i + 2 - beginIdx);
                    i = beginIdx;
                    if (trueMultLineComm) {
                        code.insert(i, LINE_ENDING);
                        ++i;
                    }
                    trueMultLineComm = false;
                    inMultLineComm = false;
                }
            }
        } else {
            if (code[i] == '/' && code[i + 1] == '/') {
                beginIdx = i;
                inSingLineComm = true;
            } else if (code[i] == '/' && code[i + 1] == '*') {
                beginIdx = i;
                inMultLineComm = true;
            }
        }
    }
}

void Parser::ProcessLineTransfer(std::string& code) {

    for (auto i = code.begin(); i < code.end();) {
        if (*i == '\\' && *(i + 1) == '\n') {
            i = code.erase(i); // remove back-slash
            i = code.erase(i); // remove newline
        } else {
            ++i;
        }
    }
}

Parser::PreprocessData Parser::ConvertCode(std::string& code) {

    // Hack: wrap all {} symbols inside a ; to treat them as instruction
    std::string symbList = "{}";
    const auto separator = ';';
    for (auto i = code.begin(); i < code.end(); ++i) {
        if (symbList.find(*i) != std::string::npos) {
            i = code.insert(i, separator) + 1;
            i = code.insert(i + 1, separator);
        }
    }
    // Hack: make preprocess instructions separator-terminted
    for (auto i = code.begin();; ++i) {
        i = SkipSpaces(i, code.end());
        bool needClose = false;
        for (auto dir : m_FreeDirictives) {
            if (ContainsFrom(i, code.end(), dir)) {
                needClose = true;
                break;
            }
        }
        i = std::find(i, code.end(), '\n');
        if (needClose)
            i = code.insert(i, separator) + 1;
        if (i == code.end())
            break;
    }

    Parser::PreprocessData ret;
    for (auto i = code.begin();; ++i) {
        auto start = i = SkipSpaces(i, code.end());
        i = std::find(i, code.end(), separator);
        auto end = SkipSpaces(std::string::reverse_iterator(i), code.rend()).base();
        if (start == code.end())
            break;
        if (start < end)
            ret.push_back(std::string{ start, end });
    }

    return ret;
}

Data::RawData Parser::ConvertCode(const PreprocessData& code) {

    return {code.begin(), code.end()};
}

void Parser::PreprocessCode(PreprocessData& code) const {

    // @todo refactoring: use SmartIterator

    // Parce properties' directives
    for (auto iter = code.begin(); iter != code.end();) {
        // @todo refactoring: move parsers into Dirictives class
        if (iter->find(Dirictives::VERSION) != std::string::npos) {
            auto i = std::find_if(iter->begin(), iter->end(), [](const char c) {
                return std::isdigit(c);
            });
            if (i == iter->end())
                continue;
            size_t startIdx = i - iter->begin();

            size_t dotIdx = iter->find(".", 9);
            if (dotIdx == std::string::npos)
                continue;

            auto ri = std::find_if(iter->rbegin(), iter->rend(), [](const char c) {
                return std::isdigit(c);
            });
            if (ri == iter->rend())
                continue;
            size_t endIdx = iter->rend() - ri;

            int8_t versionMajor = std::atoi(iter->substr(startIdx, dotIdx - startIdx).c_str());
            int8_t versionMinor = std::atoi(iter->substr(dotIdx + 1, endIdx - dotIdx - 1).c_str());
            m_PtxProps.version = { versionMajor, versionMinor };
            iter = code.erase(iter);
        } else if (iter->find(Dirictives::TARGET) != std::string::npos) {
            size_t delimIdx = iter->find("_", 9);
            if (delimIdx == std::string::npos)
                break;

            auto ri = std::find_if(iter->rbegin(), iter->rend(), [](const char c) {
                return std::isdigit(c);
            });
            if (ri == iter->rend())
                continue;
            size_t endIdx = iter->rend() - ri;

            m_PtxProps.target = std::atoi(iter->substr(delimIdx + 1, endIdx - delimIdx - 1).c_str());
            iter = code.erase(iter);
        } else if (iter->find(Dirictives::ADDRESS_SIZE) != std::string::npos) {
            auto i = std::find_if(iter->begin(), iter->end(), [](const char c) {
                return std::isdigit(c);
            });
            if (i == iter->end())
                continue;
            size_t startIdx = i - iter->begin();

            auto ri = std::find_if(iter->rbegin(), iter->rend(), [](const char c) {
                return std::isdigit(c);
            });
            if (ri == iter->rend())
                continue;
            size_t endIdx = iter->rend() - ri;

            m_PtxProps.addressSize = std::atoi(iter->substr(startIdx, startIdx - endIdx).c_str());
            iter = code.erase(iter);
        } else {
            ++iter;
        }

        if (m_PtxProps.IsValid())
            break;
    }

    m_State = State::Preprocessed;
}

bool Parser::InitVTable() {

    using namespace StringIteration;

    if (m_State != State::Preprocessed) {
        PRINT_E("Initing a virtual functions table for non-preprocessed code");
        return false;
    }

    // Parse functions
    for (; m_DataIter.IsValid(); ++m_DataIter) {

        const auto& line = *m_DataIter.Current();
        SmartIterator lineIter{line};

        bool isFunc = false;
        std::string buf;
        while (!(buf = lineIter.ReadWord()).empty()) {
            if (IsFuncDefDirictive(buf)) {
                isFunc = true;
                break;
            }
        }

        if (!isFunc)
            continue;

        Types::Function func;

        // get returns, name and arguments
        lineIter.GoToNextNonSpace();
        lineIter.EnterBracket();
        // we're now located at eigther returns or name (due to ended on a special dirictive)
        if (lineIter.IsInBracket()) {
            // return (it is single)
            auto startIter = lineIter.Current();
            auto endIter   = lineIter.ExitBracket();
            func.returns.emplace(ParsePtxVar({startIter, endIter}));
        }
        // now it is the name
        func.name = lineIter.ReadWord();
        // and right after the name we got the arguments
        lineIter.GoToNextNonSpace();
        lineIter.EnterBracket();
        // arguments
        while (lineIter.IsInBracket() && lineIter.IsValid()) {
            auto argStr = lineIter.ReadWord(false, Brackets | Punct);
            func.arguments.emplace(ParsePtxVar(argStr));
            lineIter.Skip(AllSpaces | Brackets);
        }

        // parse attributes
        lineIter.Reset();
        for (;;) {
            if (lineIter.IsBracket()) {
                lineIter.ExitBracket();
            }
            buf = lineIter.ReadWord();
            if (buf.empty()) {
                break;
            }
            if (buf.front() == '.') {
                auto& attribute = func.attributes[buf];
                buf = lineIter.ReadWord(true);
                if (buf.front() != '.') {
                    attribute = buf;
                }
            }
        }

        // Next to the function declaration we eigther get a body or nothing
        ++m_DataIter;
        if (m_DataIter.IsBlockStart()) {
            ++m_DataIter;
            // We are inside the body
            func.InsertInstructions(m_DataIter);
            if (!m_DataIter.IsBlockEnd()) {
                PRINT_E("Function is not fully initialized. "
                        "Intructions list is not complete.");
            }
            m_DataIter.ExitBlock();
        }

        auto funcFound = std::find_if(m_FuncsList.begin(), m_FuncsList.end(),
            [&func](const Types::Function& f) {
                return (f.name == func.name);
            });
        if (funcFound != m_FuncsList.end()) {
            // @todo implementation: maybe better to merge values, not remove olds
            m_FuncsList.erase(std::move(funcFound));
        }
        m_FuncsList.push_back(std::move(func));
    }

    return true;
}

Types::FuncsList::iterator
Parser::FindFunction(const std::string& funcName,
                     const Types::PTXVarList& arguments) const {

    return std::find_if(m_FuncsList.begin(), m_FuncsList.end(),
        [&](const Types::FuncsList::value_type& func) {
            // is entry
            if (!IsKernelFunction(func)) {
                return false;
            }
            // correct name
            if (func.name != funcName) {
                return false;
            }
            // correct arguments
            if (func.arguments.size() != arguments.size()) {
                return false;
            }
            Types::PTXVarList::size_type i = 0;
            for (auto& arg : func.arguments) {
                if (!arguments[i] ||
                    arg.second.type != arguments[i]->GetPTXType()) {
                    return false;
                }
                ++i;
            }
            return true;
        });
}

bool Parser::IsFuncDefDirictive(const std::string &dirictive) {

    return (std::find(m_FuncDefDirictives.begin(),
                      m_FuncDefDirictives.end(), dirictive)
            != m_FuncDefDirictives.end());
}


bool Parser::IsKernelFunction(const Types::Function& function) {

    return
        function.attributes.contains(KernelAttributes::ENTRY);
}
