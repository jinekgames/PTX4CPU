#include "parser_data.h"


namespace PTX4CPU {
namespace Data {

Iterator::Iterator() {
    Reset();
}

Iterator::Iterator(P::RawType&& data) : m_Data(new P::RawType(std::move(data))) {
    Reset();
}

Iterator::Iterator(const Iterator& right) {
    Copy(*this, right);
}

Iterator::Iterator(Iterator&& right) {
    Move(*this, right);
}

Iterator& Iterator::operator = (const Iterator& right) {
    Copy(*this, right);
    return *this;
}

Iterator& Iterator::operator = (Iterator&& right) {
    Move(*this, right);
    return *this;
}

void Iterator::Copy(Iterator& left, const Iterator& right) {
    if (&left == &right)
        return;

    left.m_Data = right.m_Data;
    left.Reset();
    left.Shift(right.Offset());
}

void Iterator::Move(Iterator& left, Iterator& right) {
    if (&left == &right)
        return;

    left.m_CurIter = std::move(right.m_CurIter);
    left.m_Data = std::move(right.m_Data);
    right.Reset();
}

const Iterator::RawType& Iterator::Data() const {
    return *m_Data;
}

const Iterator::IterType Iterator::Begin() const {
    return m_Data->begin();
}

const Iterator::IterType Iterator::End() const {
    return m_Data->end();
}

const Iterator::IterType Iterator::Next() const {
    if (IsBlockStart())
        ++m_BlocksFallCount;
    else if (IsBlockEnd())
        --m_BlocksFallCount;
    if (IsValid())
        ++m_CurIter;
    return m_CurIter;
}

const Iterator::IterType Iterator::Prev() const {
    if (IsBlockStart())
        --m_BlocksFallCount;
    else if (IsBlockEnd())
        ++m_BlocksFallCount;
    if (IsValid())
        --m_CurIter;
    return m_CurIter;
}

bool Iterator::IsInBlock() const {
    return m_BlocksFallCount;
}

bool Iterator::IsBlockStart() const {
    return !IsValid() || (*m_CurIter == OPEN_BLOCK);
}

bool Iterator::IsBlockEnd() const {
    return !IsValid() || (*m_CurIter == CLOSE_BLOCK);
}

const Iterator::IterType Iterator::ExitBlock() const {
    if (!IsInBlock())
        return m_CurIter;
    const auto targetLvl = GetBlockDepth() - 1;
    while (IsValid() && GetBlockDepth() != targetLvl)
        Next();
    return m_CurIter;
}

int64_t Iterator::GetBlockDepth() const {
    return m_BlocksFallCount;
}

bool Iterator::IsValid() const {
    return m_Data && P::IsValid();
}

const Iterator::IterType Iterator::Reset() const {
    if (m_Data)
        m_CurIter = m_Data->begin();
    else
        m_CurIter = {};
    m_BlocksFallCount = 0;
    return m_CurIter;
}

const std::string& Iterator::ReadInstruction() const {
    return *m_CurIter;
}

}  // namespace Data
}  // namespace PTX4CPU
