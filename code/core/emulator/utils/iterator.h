#pragma once

#include <type_traits>


namespace BaseTypes {

template<class T>
class IteratorBase {

public:

    using SizeType = T::size_type;
    using RawType  = std::remove_cvref_t<T>;
    using IterType = std::conditional<std::is_const_v<T>,
                                      typename T::const_iterator,
                                      typename T::iterator>::type;
    using ConstIterType = T::const_iterator;

public:

    // Returns the iterating object
    virtual const RawType& Data() const = 0;

    // Returns the iterator to the beginning of the iterating object
    virtual const IterType Begin() const = 0;
    // Returns the iterator to the end of the iterating object
    virtual const IterType End()   const = 0;

    // Returns the current iterator
    const IterType Current() const;

    // Move interator to the following element
    // Returns the current iterator
    virtual const IterType Next() const = 0;
    // Move interator to the previous element
    // Returns the current iterator
    virtual const IterType Prev() const = 0;

    IteratorBase& operator ++ ();
    const IteratorBase& operator ++ () const;
    IteratorBase& operator -- ();
    const IteratorBase& operator -- () const;

    // Returns the offset between the current iterator and the beginning of the
    // iterating object
    SizeType Offset() const;

    // Checks if the iterator is in valid state
    virtual bool IsValid() const;

    // Shifts the iterator to the specified offset
    // The offset could be above or lower the zero
    const IterType Shift(int64_t offset) const;

    // Resets the current interator to the beginning of the iterating object
    virtual const IterType Reset() const = 0;

protected:

    // Current iterator
    mutable IterType m_CurIter;

};  // class IteratorBase

}  // namespace BaseTypes


#include "iterator.inl"
