#pragma once

#include <string>
#include <vector>


namespace PTX2ASM {
namespace ParserInternal {

// Interating code type (base work) (each instruction into one line)
using Data           = std::vector<std::string>;

// @todo refactoring: move realizations into a separate file and add comments
// @todo refactoring: make a parent base class for all iterators
class DataIterator {

public:

    using Iter      = Data::iterator;
    using ConstIter = Data::const_iterator;

    using Size = Data::size_type;
    inline static const Size Npos = static_cast<Size>(-1l);

    DataIterator() : m_Data{} {
        Reset();
    }
    DataIterator(Data&& data) : m_Data(data) {
        Reset();
    }
    DataIterator(const DataIterator&) = delete;
    DataIterator(DataIterator&& right)
        : m_Data            (right.m_Data)
        , m_CurIter         (right.m_CurIter)
        , m_BlocksFallCount (right.m_BlocksFallCount) {

        right.m_Data            = {};
        right.m_CurIter         = right.m_Data.begin();
        right.m_BlocksFallCount = 0;
    }
    ~DataIterator() = default;
    DataIterator& operator = (const DataIterator&) = delete;
    DataIterator& operator = (DataIterator&& right) {
        if (&right == this)
            return *this;

        m_Data            = std::move(right.m_Data);
        m_CurIter         = right.m_CurIter;
        m_BlocksFallCount = right.m_BlocksFallCount;

        right.m_CurIter         = right.m_Data.begin();
        right.m_BlocksFallCount = 0;

        return *this;
    }

    void SwapData(Data&& data) {
        m_Data = data;
        Reset();
    }

    const Data& GetData() const { return m_Data; }

    ConstIter Begin() const { return m_Data.begin(); }
    ConstIter End()   const { return m_Data.end(); }

    ConstIter Next()  const {
        if (IsBlockStart())
            ++m_BlocksFallCount;
        else if (IsBlockEnd())
            --m_BlocksFallCount;
        if (IsValid())
            ++m_CurIter;
        return m_CurIter;
    }
    ConstIter Prev() const {
        if (IsBlockStart())
            --m_BlocksFallCount;
        else if (IsBlockEnd())
            ++m_BlocksFallCount;
        if (IsValid())
            --m_CurIter;
        return m_CurIter;
    }

    ConstIter GetIter() { return m_CurIter; }

    Size GetOffset() { return m_CurIter - Begin(); }

    bool IsInBlock() const { return m_BlocksFallCount; }

    bool IsBlockStart() const { return !IsValid() || (*m_CurIter == OPEN_BLOCK); }
    bool IsBlockEnd()   const { return !IsValid() || (*m_CurIter == CLOSE_BLOCK); }

    ConstIter ExitBlock() const {
        if (!IsInBlock())
            return m_CurIter;
        const auto targetLvl = GetBracketLvl() - 1;
        while (IsValid() && GetBracketLvl() != targetLvl)
            Next();
        return m_CurIter;
    }

    int64_t GetBracketLvl() const { return m_BlocksFallCount; }

    bool IsValid() const {
        return m_CurIter >= Begin() && m_CurIter < End();
    }

    ConstIter Shift(int64_t offset) const {
        auto absOffset = std::abs(offset);
        for (int64_t i = 0; IsValid() && i < absOffset; ++i) {
            if (offset > 0)
                Next();
            else
                Prev();
        }
        return m_CurIter;
    }

    ConstIter Reset() {
        m_CurIter = m_Data.begin();
        m_BlocksFallCount = 0;
        return m_CurIter;
    }

private:

    Data m_Data;

    mutable Iter m_CurIter;

    // Level of block inside which we are
    mutable int64_t m_BlocksFallCount = 0;

    inline static const std::string OPEN_BLOCK = "{";
    inline static const std::string CLOSE_BLOCK = "}";
};

}  // namespace ParserInternal
}  // namespace PTX2ASM
