#pragma once

#include "iterator.h"

#include <cmath>


namespace BaseTypes {

template<class T>
const typename IteratorBase<T>::IterType
IteratorBase<T>::Current() const {
    return m_CurIter;
}

template<class T>
IteratorBase<T>& IteratorBase<T>::operator ++ () {
    Next();
    return *this;
}
template<class T>
const IteratorBase<T>& IteratorBase<T>::operator ++ () const {
    Next();
    return *this;
}
template<class T>
IteratorBase<T>& IteratorBase<T>::operator -- () {
    Prev();
    return *this;
}
template<class T>
const IteratorBase<T>& IteratorBase<T>::operator -- () const {
    Prev();
    return *this;
}

template<class T>
IteratorBase<T>::SizeType IteratorBase<T>::Offset() const {
    return Current() - Begin();
}

template<class T>
bool IteratorBase<T>::IsValid() const {
    return m_CurIter >= Begin() && m_CurIter < End();
}

template<class T>
const typename IteratorBase<T>::IterType
IteratorBase<T>::Shift(int64_t offset) const {
    auto absOffset = std::abs(offset);
    for (int64_t i = 0; IsValid() && i < absOffset; ++i) {
        if (offset > 0)
            Next();
        else
            Prev();
    }
    return m_CurIter;
}

}  // namespace BaseTypes
