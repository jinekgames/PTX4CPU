#pragma once

#include <cctype>
#include <string>


size_t NextLiteral(std::string str, size_t from) {
    // skip current element
    for(;;) {
        if (!std::isalpha(str[from])) {
            ++from;
        } else {
            break;
        }
    }
    for(;;) {
        if (std::isalpha(str[from])) {
            ++from;
        } else {
            break;
        }
    }
    return from;
}