#pragma once

#include <vector>
#include <map>
#include <unordered_map>

#include "../iterator.h"
#include "string_types.h"


namespace StringIteration {

enum class Bracket {
    No = 0,
    Circle,
    Square,
    Figure,
};

enum CloseType {
    No = 0,
    Close,
    Open,
};

using BracketType = std::pair<const Bracket, const CloseType>;

inline static const std::map<char, BracketType> bracketsTable = {
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

inline constexpr WordDelimiter operator & (WordDelimiter left, WordDelimiter right) {
    return static_cast<WordDelimiter>(static_cast<int>(left) & static_cast<int>(right));
}
inline constexpr WordDelimiter operator | (WordDelimiter left, WordDelimiter right) {
    return static_cast<WordDelimiter>(static_cast<int>(left) | static_cast<int>(right));
}

bool IsDelimiter(char symbol, WordDelimiter delimiter);

template<BaseTypes::String Str>
class SmartIterator final : public BaseTypes::IteratorBase<Str> {

private:

    using P = BaseTypes::IteratorBase<Str>;

public:

    using BracketStack = std::vector<Bracket>;

    SmartIterator() = delete;
    SmartIterator(Str& str,  P::SizeType offset = 0);
    SmartIterator(Str&& str, P::SizeType offset = 0);
    ~SmartIterator() = default;

    // Returns the iterating string
    const P::RawType& Data() const override { return m_Str; }

    // Returns the iterator to the beginning of the string
    const P::IterType Begin() const override { return m_Str.begin(); }
    // Returns the iterator to the end of the string
    const P::IterType End()   const override { return m_Str.end(); }

    // Move interator to the following element
    // Returns the current iterator
    const P::IterType Next() const override;
    // Move interator to the previous element
    // Returns the current iterator
    const P::IterType Prev() const override;

    const P::IterType Erase();

    // Checks if `iter` is located on the bracket
    static bool IsBracket(const P::IterType& iter);
    // Returns the type of the bracke which `iter` is pointing to
    static BracketType WhichBracket(const P::IterType& iter);

    // Checks if iterator is located inside the brackets
    bool IsInBracket() const { return !m_BracketsStack.empty(); }
    // Returns the brackets type the iterator is located inside
    // If staying on a bracket, it is not counted
    Bracket GetBracket() const;
    // Returns the depth of current bracket
    BracketStack::size_type GetBracketDepth() const;

    // Checks if iterator is located on the space symbol
    bool IsSpace()   const { return std::isspace(GetChar()); }
    // Checks if iterator is located on the letter
    bool IsAlpha()   const { return std::isalpha(GetChar()); }
    // Checks if iterator is located on the upper-case letter
    bool IsUpper()   const { return std::isupper(GetChar()); }
    // Checks if iterator is located on the lower-case letter
    bool IsLower()   const { return std::islower(GetChar()); }
    // Checks if iterator is located on the digit
    bool IsDigit()   const { return std::isdigit(GetChar()); }
    // Checks if iterator is located on the bracket
    bool IsBracket() const { return IsBracket(P::m_CurIter); }

    // Resets the current interator to the beginning of the string
    const P::IterType Reset() const override;

    // Moves iterator to the first following non-space symbol
    // If current location is not pointing to the space symbol, return current
    const P::IterType GoToNextNonSpace() const;

    // Moves iterator to the first symbol after the currently opened bracket
    // If current location is not pointing to the bracket, return current
    // If `reverse`, it will go backward
    // If `recursive`, will move iterator throw several brackets in a row
    const P::IterType EnterBracket(bool reverse = false,
                                   bool recursive = false) const;

    // Moves iterator to the first symbol outside of the currently entered
    // brackets range
    // If `enterFirst` and standing on an opening bracket, it would be entered
    // and the pointer will be moved to the first outside of this brackets depth
    // In the other case return current
    const P::IterType ExitBracket(bool enterFirst = true) const;

    // Moves iterator to the first found specified `delimiter`
    const P::IterType GoTo(WordDelimiter delimiter) const;
    // Moves iterator to the first symbol after the specified `delimiter`
    // If several such delimiters are in a row, all of them will be skipped
    const P::IterType Skip(WordDelimiter delimiter) const;

    // The core. Reads the word starting from current position and ended on the
    // `delimiter`
    // Moves iterator to the first symbol outside of the word
    // Returns the word as string
    // If `keepLocation` the pointer will be shifted back to start position
    P::RawType ReadWord(bool keepLocation = false,
                        WordDelimiter delimiter = CodeDelimiter) const;

    // Returns current char.
    P::RawType::value_type GetChar() const { return *P::m_CurIter; }

private:

    // Iterating string
    Str& m_Str;

    mutable BracketStack m_BracketsStack;

};  // class SmartIterator

}  // namespace StringIteration


#include "string_iterator.inl"
