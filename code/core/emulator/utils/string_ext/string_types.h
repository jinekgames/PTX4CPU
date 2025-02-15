#pragma once

#include <string>
#include <string_view>
#include <type_traits>


namespace BaseTypes {

template<class T>
concept String = std::disjunction_v<
    std::is_same<std::remove_cvref_t<std::remove_all_extents_t<T>>, std::string>,
    std::is_same<std::remove_cvref_t<std::remove_all_extents_t<T>>, std::string_view>>;

template<class T>
struct IsStringIter_t {
    static constexpr bool value =
        std::is_same_v<T, std::string::iterator>                ||
        std::is_same_v<T, std::string::const_iterator>          ||
        std::is_same_v<T, std::string::reverse_iterator>        ||
        std::is_same_v<T, std::string::const_reverse_iterator>;
};

template<class T>
constexpr bool IsStringIter = IsStringIter_t<T>::value;

template<class T>
concept StringIter = IsStringIter<T>;

template<class T>
struct IsReverseStringIter_t {
    static constexpr bool value =
        std::is_same_v<T, std::string::reverse_iterator>        ||
        std::is_same_v<T, std::string::const_reverse_iterator>;
};

template<class T>
constexpr bool IsReverseStringIter = IsReverseStringIter_t<T>::value;

template<class T>
concept ReverseStringIter = IsReverseStringIter<T>;

}  // namespace BaseTypes
