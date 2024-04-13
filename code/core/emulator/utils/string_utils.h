#pragma once

#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <type_traits>

const std::string LINE_ENDING =
#if defined(WIN32)
    "\n"; // "\r\n" is not used in PTX;
#else
    "\n";
#endif

// @todo refactoring: move concepts into the special file

template<class T>
concept String = std::is_same_v<std::remove_all_extents_t<T>, std::string>;
template<class T>
concept StringIter = std::is_same_v<T, std::string::iterator>               ||
                     std::is_same_v<T, std::string::const_iterator>         ||
                     std::is_same_v<T, std::string::reverse_iterator>       ||
                     std::is_same_v<T, std::string::const_reverse_iterator>;
template<class T>
concept ReverseStringIter = std::is_same_v<T, std::string::reverse_iterator>       ||
                            std::is_same_v<T, std::string::const_reverse_iterator>;

/**
 * Returns iterator pointed to the first non-space symbol after the given one
*/
template<class StringIter>
StringIter SkipSpaces(const StringIter& iter, const StringIter& end) {

    auto ret = iter;
    while (ret < end && std::isspace(*ret))
        ++ret;
    return ret;
}

/**
 * Returns iterator pointed to the first space symbol after the given one
*/
template<class StringIter>
StringIter FindSpace(const StringIter& iter, const StringIter& end) {

    auto ret = iter;
    while (ret < end && !std::isspace(*ret))
        ++ret;
    return ret;
}

/**
 * Checks if string contains the `comp` from the `from`
*/
template<class StringIter>
bool ContainsFrom(const StringIter& from, const StringIter& end, const std::string& comp) {

    auto compIter = comp.begin();
    for (auto sourceIter = from;
         sourceIter < end && compIter < comp.end();
         ++sourceIter, ++compIter) {
        if(*sourceIter != *compIter)
            return false;
    }
    return compIter == comp.end();
}

/**
 * Splits a sting with a delimiter
*/
template<class String>
std::vector<String> Split(const String& str, char delimiter) {

  std::vector<std::string> ret;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    ret.push_back(std::move(item));
  }
  return ret;
}

// @todo refactoring: move following into the speciel files and add comments

namespace StringIteration {

enum class Bracket {
    No = 0,
    Circle,
    Square,
    Figure,
};

enum CloseType {
    Close,
    Open
};

inline static const std::map<char, std::pair<Bracket, CloseType>> bracketsTable = {
    { '(', { Bracket::Circle, Open  } },
    { ')', { Bracket::Circle, Close } },
    { '[', { Bracket::Square, Open  } },
    { ']', { Bracket::Square, Close } },
    { '{', { Bracket::Figure, Open  } },
    { '}', { Bracket::Figure, Close } },
};

enum WordDelimiter {

    NoDelim       = 0,

    // Base types

    // Any space symbol (w/out newlines)
    Space         = 1,
    // New lines
    NewLine       = 1 << 1,
    // Punct/Colon/Semicolon
    Punct         = 1 << 2,
    // Dot symbol
    Dot           = 1 << 3,
    // Any bracket
    Brackets      = 1 << 4,
    // Math operators
    MathOperators = 1 << 5,
    // Backward slash
    BackSlash     = 1 << 6,

    // Predefined combinations

    // All spaces (basic spaces + new lines)
    AllSpaces          = Space | NewLine,
    // Default for code (excluding a dot, cos it is a part of a number and dirictives)
    CodeDelimiter      = AllSpaces | Punct | Brackets | MathOperators,
    // For basic text iteration (all non-alpha symbols)
    AllTextPunctuation = CodeDelimiter | Dot | BackSlash,
    // Any known delimiter
    Any                = ~NoDelim,

};

// Base delim type to it's possible chars
inline static const std::unordered_map<WordDelimiter, std::string> baseDelimsTable = {

    { Space,         " \t" },
    { NewLine,       "\n\r" },
    { Punct,         ",:;" },
    { Dot,           "." },
    { Brackets,      "()[]{}" },
    { MathOperators, "+-/*^><=" },
    { BackSlash,     "\\" },

};

inline WordDelimiter operator & (WordDelimiter left, WordDelimiter right) {
    return static_cast<WordDelimiter>(static_cast<int>(left) & static_cast<int>(right));
}
inline WordDelimiter operator | (WordDelimiter left, WordDelimiter right) {
    return static_cast<WordDelimiter>(static_cast<int>(left) | static_cast<int>(right));
}

inline bool IsDelimiter(char symbol, WordDelimiter delimiter) {
    for (const auto& delimData : baseDelimsTable) {
        if (delimData.first & delimiter &&
            delimData.second.find(symbol) != std::string::npos) {
                return true;
        }
    }
    return false;
}

template<class String>
class SmartIterator {

public:

    using StrType    = String;
    using SizeType   = StrType::size_type;
    using RawStrType = std::remove_cvref_t<StrType>;
    using IterType = std::conditional<std::is_const_v<StrType>,
                                      typename StrType::const_iterator,
                                      typename StrType::iterator>::type;

    using BracketStack = std::vector<Bracket>;

    SmartIterator() = delete;
    // @todo refactoring: make a string rvalue
    SmartIterator(StrType& str, SizeType offset = 0)
        : m_Str{str}
        , m_CurIter{m_Str.begin() + offset}
    {}
    ~SmartIterator() = default;

    const RawStrType& GetString() const { return m_Str; }

    const IterType Begin() const { return m_Str.begin(); }
    const IterType End()   const { return m_Str.end(); }

    const IterType Next() const {
        if (IsBracket(m_CurIter)) {
            auto bracket = bracketsTable.at(*m_CurIter);
            constexpr auto openType = ReverseStringIter<IterType> ? Close : Open;
            if (bracket.second == openType)
                m_BracketsStack.push_back(bracket.first);
            else if (!m_BracketsStack.empty())
                m_BracketsStack.pop_back();
        }
        if(IsValid())
            ++m_CurIter;
        return m_CurIter;
    }
    const IterType Prev() const {
        if (IsBracket(m_CurIter)) {
            auto bracket = bracketsTable.at(*m_CurIter);
            constexpr auto openType = ReverseStringIter<IterType> ? Open : Close;
            if (bracket.second == openType)
                m_BracketsStack.push_back(bracket.first);
            else if (!m_BracketsStack.empty())
                m_BracketsStack.pop_back();
        }
        if(IsValid())
            --m_CurIter;
        return m_CurIter;
    }

    const IterType Erase() {
        m_CurIter = m_Str.erase(m_CurIter);
        return m_CurIter;
    }

    const IterType GetIter() const {
        return m_CurIter;
    }

    SizeType GetOffset() const { return m_CurIter - Begin(); }

    static bool IsBracket(const IterType& iter) {
        return bracketsTable.find(*iter) != bracketsTable.end();
    }

    bool IsInBracket() const { return !m_BracketsStack.empty(); }
    Bracket GetBracket() const {
        if (m_BracketsStack.empty())
            return Bracket::No;
        return m_BracketsStack.back();
    }
    BracketStack::size_type GetBracketLvl() const { return m_BracketsStack.size(); }

    bool IsSpace()   const { return std::isspace(*m_CurIter); }
    bool IsAlpha()   const { return std::isalpha(*m_CurIter); }
    bool IsUpper()   const { return std::isupper(*m_CurIter); }
    bool IsLower()   const { return std::islower(*m_CurIter); }
    bool IsDigit()   const { return std::isdigit(*m_CurIter); }
    bool IsBracket() const { return IsBracket(m_CurIter); }

    bool IsValid() const {
        return m_CurIter >= Begin() && m_CurIter < End();
    }

    const IterType Shift(int64_t offset) const {
        auto absOffset = std::abs(offset);
        for (int64_t i = 0; IsValid() && i < absOffset; ++i) {
            if (offset > 0)
                Next();
            else
                Prev();
        }
        return m_CurIter;
    }

    const IterType Reset() const {
        m_CurIter = Begin();
        m_BracketsStack.clear();
        return m_CurIter;
    }

    // @todo refactoring: delete depricated

    // depricated
    const IterType GoToNextSpace() const {
        while(IsValid() && !IsSpace())
            Next();
        return m_CurIter;
    }
    // depricated
    const IterType GoToNextNonSpace() const {
        while(IsValid() && IsSpace())
            Next();
        return m_CurIter;
    }

    // depricated
    const IterType EnterBracket(bool reverse = false, bool recursive = false) const {
        while (IsValid() && IsBracket(m_CurIter)) {
            if (reverse)
                Prev();
            else
                Next();
            if(!recursive)
                break;
        }
        return m_CurIter;
    }

    const IterType ExitBracket() const {
        auto startIter = m_CurIter;
        if (IsBracket())
            EnterBracket();
        if (!IsInBracket())
            return Shift(m_CurIter - startIter);
        const auto targetLvl = GetBracketLvl() - 1;
        while (IsValid() && GetBracketLvl() != targetLvl)
            Next();
        return m_CurIter;
    }

    // depricated
    RawStrType ReadWord(bool keepLocation = false, bool ignoreBrackets = true) const {
        RawStrType ret;
        auto oldIter = m_CurIter;
        GoToNextNonSpace();
        if (ignoreBrackets)
            EnterBracket(false, true);
        auto startIter = m_CurIter;
        GoToNextSpace();
        if (ignoreBrackets)
            EnterBracket(true, true);
        auto endIter = m_CurIter;
        if (oldIter < m_CurIter && IsValid())
            ret = RawStrType{startIter, endIter};
        if (keepLocation)
            Shift(oldIter - m_CurIter);
        return ret;
    }

    const IterType GoTo(WordDelimiter delimiter) const {
        while(IsValid() && !IsDelimiter(*m_CurIter, delimiter))
            Next();
        return m_CurIter;
    }
    const IterType Skip(WordDelimiter delimiter) const {
        while(IsValid() && IsDelimiter(*m_CurIter, delimiter))
            Next();
        return m_CurIter;
    }

    RawStrType ReadWord2(bool keepLocation = false,
                         WordDelimiter delimiter = CodeDelimiter) const {

        RawStrType ret;
        auto oldIter = m_CurIter;
        Skip(delimiter);
        auto startIter = m_CurIter;
        GoTo(delimiter);
        auto endIter = m_CurIter;
        if (oldIter < m_CurIter)
            ret = {startIter, endIter};
        if (keepLocation)
            Shift(oldIter - m_CurIter);
        return ret;
    }

private:

    StrType& m_Str;

    mutable IterType m_CurIter;

    mutable BracketStack m_BracketsStack;

}; // class SmartIterator

} // namespace StringIteration

#define CONCAT_INTERNAL(first, second) \
    first##second

#define CONCAT(first, second) \
    CONCAT_INTERNAL(first, second)
