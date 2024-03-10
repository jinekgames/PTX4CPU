#include <parser.h>

#include <logger.h>
#include <string_utils.h>


namespace PTX2ASM {

// Constructors and Destructors

Parser::Parser(const std::string& source) {

    auto res = Load(source);
}


// Private realizations

Result Parser::Load(const std::string& source) {

    auto rawCode = source;
    // @todo optimization: replace with list of chars
    // @todo optimization: replace string references with pointers
    //                     to empty memory then
    // @todo implementation: add logging on non-supported dirictives

    ProcessLineTransfer(rawCode);
    ClearCodeComments(rawCode);
    auto code = ConvertCode(rawCode);

    PreprocessCode(code);
    m_Data = ConvertCode(code);

    if (!m_PtxProps.IsValid()) {
        PRINT_E("Missing PTX properties");
    }
    PRINT_I("PTX props: ver %d.%d sm_%d address size %d",
            m_PtxProps.version.first, m_PtxProps.version.second,
            m_PtxProps.target, m_PtxProps.addressSize);

    InitVTable();

    return ResultCode::Fail;
}

void Parser::ClearCodeComments(std::string& code) const {

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

void Parser::ProcessLineTransfer(std::string& code) const {

    for (auto i = code.begin(); i < code.end();) {
        if (*i == '\\' && *(i + 1) == '\n') {
            i = code.erase(i); // remove back-slash
            i = code.erase(i); // remove newline
        } else {
            ++i;
        }
    }
}

Parser::PreprocessData Parser::ConvertCode(std::string& code) const {

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

Parser::Data Parser::ConvertCode(const PreprocessData& code) const {

    return {code.begin(), code.end()};
}

void Parser::PreprocessCode(PreprocessData& code) const {

    // Parce properties' directives
    for (auto iter = code.begin(); iter != code.end();) {
        if (iter->find(".version") != std::string::npos) {
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
        } else if (iter->find(".target") != std::string::npos) {
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
        } else if (iter->find(".address_size") != std::string::npos) {
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

void Parser::InitVTable() {

    // Parse functions
    for (auto iter = m_Data.begin(); iter < m_Data.end(); ++iter) {

        auto& line = *iter;
        const SmartIterator lineIter{line};

        bool isFunc = false;
        std::string buf;
        while(!(buf = lineIter.ExtractWord()).empty()) {
            if (std::find(m_FuncDefDirictives.begin(), m_FuncDefDirictives.end(), buf) != m_FuncDefDirictives.end()) {
                isFunc = true;
                break;
            }
        }

        if (!isFunc)
            continue;

        Function func;

        // get returns, name and arguments
        lineIter.GoToNextNonSpace();
        lineIter.EnterBracket();
        // we're now located at eigther returns or name (due to ended on a special dirictive)
        if (lineIter.IsInBracket()) {
            // returns
            auto startIter = lineIter.GetIter();
            auto endIter   = lineIter.ExitBracket();
            std::string name;
            VarPtxType type;
            std::tie(name, type) = ParsePtxVar({startIter, endIter});
            func.returns.emplace(name, type);
        }
        // now it is the name
        func.name = lineIter.ExtractWord();
        // and right after the name we got the arguments
        lineIter.GoToNextNonSpace();
        lineIter.EnterBracket();
        if (lineIter.IsInBracket()) {
            // arguments
            auto startIter = lineIter.GetIter();
            auto endIter   = lineIter.ExitBracket();
            auto args = Split({startIter, endIter}, ',');
            for (auto arg : args) {
                std::string name;
                VarPtxType type;
                std::tie(name, type) = ParsePtxVar(arg);
                func.arguments.emplace(name, type);
            }
        }

        // parse dirictives
        lineIter.Reset();
        for(;;) {
            if (lineIter.IsInBracket())
                lineIter.ExitBracket();
            buf = lineIter.ExtractWord();
            if (buf.empty())
                break;
            if (buf.front() == '.') {
                auto& attribute = func.attributes[buf];
                buf = lineIter.ExtractWord(true);
                if (buf.front() != '.')
                    attribute = buf;
            }
        }
    }
}

std::tuple<std::string, Parser::VarPtxType> Parser::ParsePtxVar(const std::string& entry) {

    std::string name;
    Parser::VarPtxType type;
    // Sample string:
    // .reg .f64 dbl
    const SmartIterator iter{entry};
    type.attributes.push_back(iter.ExtractWord()); // contains memory location only
    type.type = iter.ExtractWord();
    name = iter.ExtractWord();
    return std::make_tuple(name, type);
}


};  // namespace PTX2ASM
