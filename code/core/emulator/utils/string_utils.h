#pragma once

#include <string>
#include <type_traits>

const std::string LINE_ENDING =
#if defined(WIN32)
    "\n"; // doesn't work for unkwon reason "\r\n";
#else
    "\n";
#endif

template<class T>
concept StringIter = std::is_same_v<T, std::string::iterator>               ||
                     std::is_same_v<T, std::string::const_iterator>         ||
                     std::is_same_v<T, std::string::reverse_iterator>       ||
                     std::is_same_v<T, std::string::const_reverse_iterator>;

/**
 * Returns iterator pointed to the first non-space symbol after the given one
*/
template<class StringIter>
StringIter SkipSpaces(const StringIter& iter, const StringIter& end) {

    auto ret = iter;
    while (ret < end && std::isspace(*ret))
        ++ret;
    return ret;
}

/**
 * Checks if string contains the `comp` from the `from`
*/
template<class StringIter>
bool ContainsFrom(const StringIter& from, const StringIter& end, const std::string& comp) {

    auto compIter = comp.begin();
    for (auto sourceIter = from;
         sourceIter < end && compIter < comp.end();
         ++sourceIter, ++compIter) {
        if(*sourceIter != *compIter)
            return false;
    }
    return compIter == comp.end();
}
