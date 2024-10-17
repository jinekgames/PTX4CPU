#pragma once

#include <list>
#include <vector>
#include <type_traits>


namespace BaseTypes {

template<class C, class T>
concept Iterable = std::is_same_v<C, std::vector<T>> ||
                   std::is_same_v<C, std::list<T>>;

}  // namespace BaseTypes
