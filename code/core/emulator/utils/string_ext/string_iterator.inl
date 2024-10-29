#pragma once

#include "string_iterator.h"


namespace StringIteration {

template<BaseTypes::String Str>
SmartIterator<Str>::SmartIterator(Str& str, P::SizeType offset)
    : m_Str{str} {

    static_assert(std::is_const_v<Str>, "Can only be used with const strings");

    P::m_CurIter = {m_Str.begin() + offset};
}

template<BaseTypes::String Str>
SmartIterator<Str>::SmartIterator(Str&& str, P::SizeType offset)
    : m_Str{str} {

    P::m_CurIter = {m_Str.begin() + offset};
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::Next() const {

    using Iter = SmartIterator<Str>::P::IterType;

    if (IsBracket(P::m_CurIter)) {
        auto bracket = bracketsTable.at(*P::m_CurIter);
        constexpr auto openType = BaseTypes::IsReverseStringIter<Iter>
                                  ? Close : Open;
        if (bracket.second == openType) {
            m_BracketsStack.push_back(bracket.first);
        }
        else if (!m_BracketsStack.empty()) {
            m_BracketsStack.pop_back();
        }
    }

    if(P::IsValid()) {
        ++P::m_CurIter;
    }
    return P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::Prev() const {

    using Iter = SmartIterator<Str>::P::IterType;

    if(P::m_CurIter <= Begin()) {
        P::m_CurIter;
    }

    if ((P::m_CurIter < End()) && IsBracket(P::m_CurIter)) {
        auto bracket = bracketsTable.at(*P::m_CurIter);
        constexpr auto openType = BaseTypes::IsReverseStringIter<Iter>
                                  ? Open : Close;
        if (bracket.second == openType) {
            m_BracketsStack.push_back(bracket.first);
        }
        else if (!m_BracketsStack.empty()) {
            m_BracketsStack.pop_back();
        }
    }

    return --P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::Erase() {

    P::m_CurIter = m_Str.erase(P::m_CurIter);
    return P::m_CurIter;
}

template<BaseTypes::String Str>
bool SmartIterator<Str>::IsBracket(const typename P::IterType& iter) {

    return bracketsTable.find(*iter) != bracketsTable.end();
}

template<BaseTypes::String Str>
BracketType SmartIterator<Str>::WhichBracket(const typename P::IterType& iter) {

    auto found = bracketsTable.find(*iter);
    if (found != bracketsTable.end()) {
        return found->second;
    }
    return {};
}

template<BaseTypes::String Str>
Bracket SmartIterator<Str>::GetBracket() const {

    if (m_BracketsStack.empty()) {
        return Bracket::No;
    }
    return m_BracketsStack.back();
}

template<BaseTypes::String Str>
SmartIterator<Str>::BracketStack::size_type
SmartIterator<Str>::GetBracketDepth() const {

    return m_BracketsStack.size();
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::Reset() const {

    P::m_CurIter = Begin();
    m_BracketsStack.clear();
    return P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::GoToNextNonSpace() const {

    while (P::IsValid() && IsSpace()) {
        Next();
    }
    return P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::EnterBracket(bool reverse, bool recursive) const {

    using Iter = SmartIterator<Str>::P::IterType;

    auto CheckBracket = [=](const Iter& iter) {
        const auto openType =
            BaseTypes::IsReverseStringIter<Iter> && reverse
            ? Close : Open;
        return WhichBracket(iter).second == openType;
    };

    while (P::IsValid() && CheckBracket(P::m_CurIter)) {
        if (reverse) {
            Prev();
        } else {
            Next();
        }
        if (!recursive) {
            break;
        }
    }

    return P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::ExitBracket(bool enterFirst) const {

    using Iter = SmartIterator<Str>::P::IterType;

    constexpr auto openType = BaseTypes::IsReverseStringIter<Iter>
                              ? Close : Open;
    if (enterFirst && (WhichBracket(P::m_CurIter).second == openType)) {
        EnterBracket();
    }
    const auto targetLvl = GetBracketDepth() - 1;
    while (P::IsValid() && GetBracketDepth() != targetLvl) {
        Next();
    }
    return P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::GoTo(WordDelimiter delimiter) const {

    while (P::IsValid() && !IsDelimiter(*P::m_CurIter, delimiter)) {
        Next();
    }
    return P::m_CurIter;
}

template<BaseTypes::String Str>
const typename SmartIterator<Str>::P::IterType
SmartIterator<Str>::Skip(WordDelimiter delimiter) const {

    while (P::IsValid() && IsDelimiter(*P::m_CurIter, delimiter)) {
        Next();
    }
    return P::m_CurIter;
}

template<BaseTypes::String Str>
SmartIterator<Str>::P::RawType
SmartIterator<Str>::ReadWord(bool keepLocation, WordDelimiter delimiter) const {

    typename P::RawType ret;
    auto oldIter = P::m_CurIter;
    Skip(delimiter);
    auto startIter = P::m_CurIter;
    GoTo(delimiter);
    auto endIter = P::m_CurIter;
    if (oldIter < P::m_CurIter) {
        ret = {startIter, endIter};
    }
    if (keepLocation) {
        P::Shift(oldIter - P::m_CurIter);
    }
    return ret;
}

}  // namespace StringIteration
