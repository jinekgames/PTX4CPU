#pragma once


namespace LogColor {

#define _LOG_COLOR_COLOR_ESCAPE "\033"

inline constexpr auto COLOR_RESET   = _LOG_COLOR_COLOR_ESCAPE "[0m";
inline constexpr auto COLOR_BLACK   = _LOG_COLOR_COLOR_ESCAPE "[30m";
inline constexpr auto COLOR_RED     = _LOG_COLOR_COLOR_ESCAPE "[31m";
inline constexpr auto COLOR_GREEN   = _LOG_COLOR_COLOR_ESCAPE "[32m";
inline constexpr auto COLOR_YELLOW  = _LOG_COLOR_COLOR_ESCAPE "[33m";
inline constexpr auto COLOR_BLUE    = _LOG_COLOR_COLOR_ESCAPE "[34m";
inline constexpr auto COLOR_MAGENTA = _LOG_COLOR_COLOR_ESCAPE "[35m";
inline constexpr auto COLOR_CYAN    = _LOG_COLOR_COLOR_ESCAPE "[36m";
inline constexpr auto COLOR_WHITE   = _LOG_COLOR_COLOR_ESCAPE "[37m";

}  // namespace LogColor
