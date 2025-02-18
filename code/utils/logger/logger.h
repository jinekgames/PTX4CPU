#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <unordered_map>

#if defined(linux)
#include <linux/limits.h>
#endif

#include "../debug.h"
#include "log_color.h"


#define LOGS_DEFAULT_TAG "PTX4CPU"

#define FORCE_DEBUG_LOGS 1
#define STRIP_DEBUG_LOGS 0

// Enables debug logs if true
#if FORCE_DEBUG_LOGS || defined(DEBUG_BUILD) && !STRIP_DEBUG_LOGS
// @todo implementation: move this into runtime config
#define LOGGER_DEBUG_LOGS 1
#else
#define LOGGER_DEBUG_LOGS 0
#endif

namespace Logs {

enum class Type {
    Debug,
    Info,
    Warning,
    Error,
};

template<Type type>
constexpr auto GetColor() {
    switch (type)
    {
    case Type::Debug:   return LogColor::COLOR_CYAN;
    case Type::Info:    return LogColor::COLOR_GREEN;
    case Type::Warning: return LogColor::COLOR_YELLOW;
    case Type::Error:   return LogColor::COLOR_RED;
    default:            break;
    }
    return LogColor::COLOR_RESET;
}

template<Type type>
constexpr auto GetTypeName() {
    switch (type)
    {
    case Type::Debug:   return "Debug";
    case Type::Info:    return "Info";
    case Type::Warning: return "Warning";
    case Type::Error:   return "Error";
    default:            break;
    }
    return LogColor::COLOR_RESET;
}

#if defined(WIN32) || defined(linux) || defined(MAC_OS)

constexpr auto SystemMaxPath() {
#if defined(WIN32)
    return _MAX_PATH;
#elif defined(linux)
    return PATH_MAX;
#else
#error "SystemMaxPath() not implemented for your platform"
#endif
}

// @todo implementation: add logging to file
// @todo implementation: add ability to separate logging files for threads

// @todo refactoring: use FormatString for formatting and C++23 API
template<Type type, class... Args>
void _app_log_message(const char* tag, const char* file, int line,
                      Args... args) {

    constexpr size_t bufSize = SystemMaxPath();
    constexpr size_t fileStrMinLen = 30;

    char buf[bufSize];
    const auto msgSize = std::snprintf(buf, bufSize, args...);

    const auto filename = std::filesystem::path(file).filename().string();

    const std::string fileStr =
        std::vformat("({}:{})", std::make_format_args(
                     filename, line));
    const std::string::size_type spacesCount = (fileStr.length() < fileStrMinLen)
                                               ? fileStrMinLen - fileStr.length()
                                               : 1u;
    const std::string spaces(spacesCount, ' ');

    const std::string_view msgStr{buf, buf + msgSize};

    static const auto colorMod = GetColor<type>();
    static const auto typeStr  = GetTypeName<type>();

    const std::string output =
        std::vformat("{}  {}{}{}{}\t: {}{}",
                     std::make_format_args(tag, fileStr, spaces,
                                           colorMod, typeStr,
                                           msgStr,
                                           LogColor::COLOR_RESET));

    std::printf("%s\n", output.c_str());

#ifdef DEBUG_BUILD

    if constexpr (type == Logs::Type::Error) {
        PTX4CPU::Debug::UnconditionalBreakpoint();
    }

#endif  // DEBUG_BUILD
}

#else
#error logs are not implemented for this platform
#endif

}  // namespace Logs

#define PRINT_TAG_E(tag, ...) Logs::_app_log_message<Logs::Type::Error>(tag, __FILE__, __LINE__, __VA_ARGS__)
#define PRINT_TAG_W(tag, ...) Logs::_app_log_message<Logs::Type::Warning>(tag, __FILE__, __LINE__, __VA_ARGS__)
#define PRINT_TAG_I(tag, ...) Logs::_app_log_message<Logs::Type::Info>(tag, __FILE__, __LINE__, __VA_ARGS__)
#if LOGGER_DEBUG_LOGS
#define PRINT_TAG_V(tag, ...) Logs::_app_log_message<Logs::Type::Debug>(tag, __FILE__, __LINE__, __VA_ARGS__)
#else
#define PRINT_TAG_V(tag, ...) do {} while(false)
#endif

#define PRINT_E(...) PRINT_TAG_E(LOGS_DEFAULT_TAG, __VA_ARGS__)
#define PRINT_W(...) PRINT_TAG_W(LOGS_DEFAULT_TAG, __VA_ARGS__)
#define PRINT_V(...) PRINT_TAG_V(LOGS_DEFAULT_TAG, __VA_ARGS__)
#define PRINT_I(...) PRINT_TAG_I(LOGS_DEFAULT_TAG, __VA_ARGS__)

#define PRINT(...) PRINT_I(__VA_ARGS__)
