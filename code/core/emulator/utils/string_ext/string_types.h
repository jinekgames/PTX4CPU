#pragma once

#include <string>
#include <type_traits>


namespace BaseTypes {

template<class T>
concept String = std::is_same_v<std::remove_cvref_t<std::remove_all_extents_t<T>>, std::string>;
template<class T>
concept StringIter = std::is_same_v<T, std::string::iterator>               ||
                     std::is_same_v<T, std::string::const_iterator>         ||
                     std::is_same_v<T, std::string::reverse_iterator>       ||
                     std::is_same_v<T, std::string::const_reverse_iterator>;
template<class T>
concept ReverseStringIter = std::is_same_v<T, std::string::reverse_iterator>       ||
                            std::is_same_v<T, std::string::const_reverse_iterator>;

}  // namespace BaseTypes
