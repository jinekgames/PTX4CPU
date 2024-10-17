#pragma once

#include <regex>
#include <string>
#include <sstream>
#include <vector>
#include <type_traits>

#include "string_ext/container_types.h"
#include "string_ext/string_iterator.h"
#include "string_ext/string_types.h"


static const std::string LINE_ENDING =
#if defined(WIN32)
    "\n"; // "\r\n" is not used in PTX;
#else
    "\n";
#endif

/**
 * Returns iterator pointed to the first non-space symbol after the given one
*/
template<BaseTypes::StringIter StringIter>
inline StringIter SkipSpaces(const StringIter& iter, const StringIter& end) {

    auto ret = iter;
    while (ret < end && std::isspace(*ret))
        ++ret;
    return ret;
}

/**
 * Returns iterator pointed to the first space symbol after the given one
*/
template<BaseTypes::StringIter StringIter>
inline StringIter FindSpace(const StringIter& iter, const StringIter& end) {

    auto ret = iter;
    while (ret < end && !std::isspace(*ret))
        ++ret;
    return ret;
}

/**
 * Checks if string contains the `comp` from the `from`
*/
template<BaseTypes::StringIter StringIter>
inline bool ContainsFrom(const StringIter& from, const StringIter& end,
                         const std::string& comp) {

    auto compIter = comp.begin();
    for (auto sourceIter = from;
         sourceIter < end && compIter < comp.end();
         ++sourceIter, ++compIter) {
        if (*sourceIter != *compIter){
            return false;
        }
    }
    return compIter == comp.end();
}

/**
 * Splits a sting with a delimiter
*/
template<BaseTypes::String String>
inline std::vector<String> Split(const String& str, char delimiter) {

  std::vector<std::string> ret;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    ret.push_back(std::move(item));
  }
  return ret;
}

/**
 * Merges a stings' list with a separator
*/
template<BaseTypes::String String, BaseTypes::Iterable<std::string> List>
inline String Merge(const String& sep, List list) {

  String ret;
  for (const auto& str : list) {
    ret += str;
    if (&str != &list.back())
    {
        ret += sep;
    }
  }
  return ret;
}

/**
 * Checks if string is a number.
 * Must match regex: (\\+|\\-|)\\d+
*/
template<BaseTypes::String String>
inline bool IsNumber(const String& str) {

  static const std::regex expr{"(\\+|\\-|)\\d+"};
  return std::regex_match(str, expr);
}


#define CONCAT_INTERNAL(first, second) \
    first##second

#define CONCAT(first, second) \
    CONCAT_INTERNAL(first, second)


#define STRINGIFY_INTERNAL(value) \
    #value

#define STRINGIFY(value) \
    STRINGIFY_INTERNAL(value)
