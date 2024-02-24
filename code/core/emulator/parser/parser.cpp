#include <parser.h>

#include <string_utils.h>


namespace PTX2ASM {

// Constructors and Destructors

Parser::Parser(const std::string& source) {

    auto res = Load(source);
}


// Private realizations

Result Parser::Load(const std::string& source) {

    auto code = source;
    // @todo optimiation: replace with list of chars

    ProcessLineTransfer(code);
    ClearCodeComments(code);
    auto preprocCode = ConvertCode(code);

    // PreprocessCode(code);
    // Parse() // @todo implement: find fucntions and store them

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

Data Parser::PreprocessCode(PreprocessData& code) const {

    // @todo implement
    // this stage should parse ptx props (copy below)
    // and process other dirictives
}

//     // Convert to Translation type
//     std::stringstream ss(m_PtxIn);
//     std::string buf;
//     m_TrIn.clear();
//     while(std::getline(ss, buf, LINE_ENDING.back())) {
//         m_TrIn.push_back(buf);
//     }

//     // Clear empty lines
//     for (size_t i = 0; i < m_TrIn.size();) {
//         bool toClear = true;
//         for (auto c : m_TrIn[i]) {
//             if (!std::isspace(c)) {
//                 toClear = false;
//                 break;
//             }
//         }
//         if (toClear) {
//             m_TrIn.erase(m_TrIn.begin() + i);
//         } else {
//             ++i;
//         }
//     }

//     // Parce properties' directives
//     for (auto iter = m_TrIn.begin(); iter != m_TrIn.end();) {
//         if (iter->find(".version") != -1) {
//             auto i = std::find_if(iter->begin(), iter->end(), [](const char c) {
//                 return std::isdigit(c);
//             });
//             if (i == iter->end())
//                 continue;
//             size_t startIdx = i - iter->begin();

//             size_t dotIdx = iter->find(".", 9);
//             if (dotIdx == -1)
//                 continue;

//             auto ri = std::find_if(iter->rbegin(), iter->rend(), [](const char c) {
//                 return std::isdigit(c);
//             });
//             if (ri == iter->rend())
//                 continue;
//             size_t endIdx = iter->rend() - ri;

//             int versionMajor = std::atoi(iter->substr(startIdx, dotIdx - startIdx).c_str());
//             int versionMinor = std::atoi(iter->substr(dotIdx + 1, endIdx - dotIdx - 1).c_str());
//             m_PtxProps.version = { versionMajor, versionMinor };
//             iter = m_TrIn.erase(iter);
//         } else if (iter->find(".target") != -1) {
//             size_t delimIdx = iter->find("_", 9);
//             if (delimIdx == -1)
//                 break;

//             auto ri = std::find_if(iter->rbegin(), iter->rend(), [](const char c) {
//                 return std::isdigit(c);
//             });
//             if (ri == iter->rend())
//                 continue;
//             size_t endIdx = iter->rend() - ri;

//             m_PtxProps.target = std::atoi(iter->substr(delimIdx + 1, endIdx - delimIdx - 1).c_str());
//             iter = m_TrIn.erase(iter);
//         } else if (iter->find(".address_size") != -1) {
//             auto i = std::find_if(iter->begin(), iter->end(), [](const char c) {
//                 return std::isdigit(c);
//             });
//             if (i == iter->end())
//                 continue;
//             size_t startIdx = i - iter->begin();

//             auto ri = std::find_if(iter->rbegin(), iter->rend(), [](const char c) {
//                 return std::isdigit(c);
//             });
//             if (ri == iter->rend())
//                 continue;
//             size_t endIdx = iter->rend() - ri;

//             m_PtxProps.addressSize = std::atoi(iter->substr(startIdx, startIdx - endIdx).c_str());
//             iter = m_TrIn.erase(iter);
//         } else {
//             ++iter;
//         }

//         if (m_PtxProps.version != PtxProperties().version &&
//             m_PtxProps.target != PtxProperties().target &&
//             m_PtxProps.addressSize != PtxProperties().addressSize)
//             break;
//     }

//     m_State = State::Preprocessed;
// }


};  // namespace PTX2ASM
