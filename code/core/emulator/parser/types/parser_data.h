#pragma once

#include <memory>
#include <string>
#include <vector>

#include <utils/iterator.h>


namespace PTX4CPU {
namespace Data {

// Interating code type (base work) (each instruction into one line)
using Type = std::vector<std::string>;

class Iterator final : public BaseTypes::IteratorBase<Type> {

private:

    using P = BaseTypes::IteratorBase<Type>;

public:

    static constexpr P::SizeType Npos = static_cast<P::SizeType>(-1l);

    Iterator();
    Iterator(P::RawType&& data);
    Iterator(const Iterator& right);
    Iterator(Iterator&& right);
    ~Iterator() = default;
    Iterator& operator = (const Iterator& right);
    Iterator& operator = (Iterator&& right);

private:

    static void Copy(Iterator& left, const Iterator& right);
    static void Move(Iterator& left, Iterator& right);

public:

    // Returns the iterating instructions list
    const P::RawType& Data() const override;

    // Returns the iterator to the beginning of the instructions list
    const P::IterType Begin() const override;
    // Returns the iterator to the end of the instructions list
    const P::IterType End()   const override;

    // Move interator to the following element
    // Returns the current iterator
    const P::IterType Next() const override;
    // Move interator to the previous element
    // Returns the current iterator
    const P::IterType Prev() const override;

    // Checks if iterator is located inside the block of code
    bool IsInBlock() const;

    // Checks if iterator is located on the opening of code block
    bool IsBlockStart() const;
    // Checks if iterator is located on the ending of code block
    bool IsBlockEnd()   const;

    // Moves iterator to the first line after the and of code block
    const P::IterType ExitBlock() const;

    // Returns the depth of current code block
    int64_t GetBlockDepth() const;

    // Checks if the iterator is in valid state
    bool IsValid() const override;

    // Resets the current interator to the beginning of the instructions list
    const P::IterType Reset() const override;

    // Returns the instruction the iterator is pointing to
    const std::string& ReadInstruction() const;

private:

    // Pointer to the iterating instructions list
    std::shared_ptr<P::RawType> m_Data;

    // Level of block inside which we are
    mutable int64_t m_BlocksFallCount = 0;

    inline static const std::string OPEN_BLOCK  = "{";
    inline static const std::string CLOSE_BLOCK = "}";
};

}  // namespace Data
}  // namespace PTX4CPU
