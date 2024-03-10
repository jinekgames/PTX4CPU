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
std::vector<std::string> Split(const std::string& str, char delimiter) ;//{

//   std::vector<std::string> ret;
//   std::stringstream ss(str);
//   std::string item;
//   while (std::getline(ss, item, delimiter)) {
//     ret.push_back(std::move(item));
//   }
//   return ret;
// }

template<class String>
class SmartIterator {

public:

    enum class Bracket {
        No,
        Circle,
        Square,
        Figure,
    };

    enum CloseType {
        Close,
        Open
    };

    typedef std::remove_all_extents_t<String> DecorFreeString;
    typedef std::conditional<std::is_const_v<String>,
                             String::const_iterator,
                             String::iterator>::type Type;
    // typedef String::iterator Type;
    typedef std::vector<Bracket> BracketStack;
    typedef String::size_type SizeType;

    SmartIterator() = delete;
    SmartIterator(String& str, SizeType offset = 0)
        : m_Str{str}
        , m_CurIter{m_Str.begin() + offset}
    {}
    ~SmartIterator() = default;

    const Type Begin() const { return m_Str.begin(); }
    const Type End()   const { return m_Str.end(); }

    const Type Next() const {
        if (IsBracket(m_CurIter)) {
            auto bracket = m_BracketTable.at(*m_CurIter);
            constexpr auto openType = ReverseStringIter<Type> ? Close : Open;
            if (bracket.second == openType)
                m_BracketsStack.push_back(bracket.first);
            else
                m_BracketsStack.pop_back();
        }
        ++m_CurIter;
        return m_CurIter;
    }
    const Type Prev() const {
        if (IsBracket(m_CurIter)) {
            auto bracket = m_BracketTable.at(*m_CurIter);
            constexpr auto openType = ReverseStringIter<Type> ? Open : Close;
            if (bracket.second == openType)
                m_BracketsStack.push_back(bracket.first);
            else
                m_BracketsStack.pop_back();
        }
        --m_CurIter;
        return m_CurIter;
    }

    const Type Erase() {
        m_CurIter = m_Str.erase(m_CurIter);
        return m_CurIter;
    }

    const Type GetIter() const {
        return m_CurIter;
    }

    SizeType GetOffset() const { return m_CurIter - Begin(); }

    static bool IsBracket(const Type& iter) {
        return m_BracketTable.find(*iter) != m_BracketTable.end();
    }

    bool IsInBracket() const { return !m_BracketsStack.empty(); }
    Bracket GetBracket() const {
        if (m_BracketsStack.empty())
            return Bracket::No;
        return m_BracketsStack.back();
    }
    BracketStack::size_type GetBracketLvl() const { return m_BracketsStack.size(); }

    bool IsSpace() const { return std::isspace(*m_CurIter); }
    bool IsAlpha() const { return std::isalpha(*m_CurIter); }
    bool IsUpper() const { return std::isupper(*m_CurIter); }
    bool IsLower() const { return std::islower(*m_CurIter); }
    bool IsDigit() const { return std::isdigit(*m_CurIter); }

    bool IsValid() const {
        return m_CurIter >= Begin() && m_CurIter < End();
    }

    const Type Shift(int64_t offset) const {
        auto adbOffset = std::abs(offset);
        for (int64_t i = 0; IsValid() && i < adbOffset; ++i) {
            if (offset > 0)
                Next();
            else
                Prev();
        }
        return m_CurIter;
    }

    const Type Reset() const {
        m_CurIter = Begin();
        m_BracketsStack.clear();
        return m_CurIter;
    }

    const Type GoToNextSpace() const {
        while(IsValid() && !IsSpace())
            Next();
        return m_CurIter;
    }
    const Type GoToNextNonSpace() const {
        while(IsValid() && IsSpace())
            Next();
        return m_CurIter;
    }

    const Type EnterBracket(bool reverse = false, bool recursive = false) const {
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

    const Type ExitBracket() const {
        if (!IsInBracket())
            return m_CurIter;
        auto targetLvl = GetBracketLvl() - 1;
        while (IsValid() && GetBracketLvl() != targetLvl)
            Next();
        return m_CurIter;
    }

    std::string ExtractWord(bool keepLocation = false, bool ignoreBrackets = true) const {
        std::string ret;
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
            ret = {startIter, endIter};
        if (keepLocation)
            Shift(oldIter - m_CurIter);
        return ret;
    }

private:

    String& m_Str;

    mutable Type m_CurIter;

    inline static const std::map<char, std::pair<Bracket, CloseType>> m_BracketTable = {
        { '(', { Bracket::Circle, Open } },
        { ')', { Bracket::Circle, Close } },
        { '[', { Bracket::Square, Open } },
        { ']', { Bracket::Square, Close } },
        { '{', { Bracket::Figure, Open } },
        { '}', { Bracket::Figure, Close } },
    };

    mutable BracketStack m_BracketsStack;

};
