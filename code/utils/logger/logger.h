#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <unordered_map>

#include "log_color.h"

#define LOGS_DEFAULT_TAG "PTX4CPU"
#define LOGS_LAYER_TAG "validation layer"

#define STRIP_DEBUG_LOGS 0

// Enables debug logs if true
#if defined(DEBUG_BUILD) && !STRIP_DEBUG_LOGS
// @todo implementation: move this into runtime config
#define DEBUG_LOGS 1
#else
#define DEBUG_LOGS 0
#endif

enum class LogType {
    Debug,
    Info,
    Warning,
    Error,
};

static std::unordered_map<LogType, const char*> logsTypeColors = {
    { LogType::Info,    COLOR_GREEN  },
    { LogType::Debug,   COLOR_CYAN   },
    { LogType::Warning, COLOR_YELLOW },
    { LogType::Error,   COLOR_RED    },
};

#if defined(WIN32) || defined(LINUX) || defined(MAC_OS)

static std::unordered_map<LogType, const char*> logsTypeStrings = {
    { LogType::Debug,   "Debug"   },
    { LogType::Info,    "Info"    },
    { LogType::Warning, "Warning" },
    { LogType::Error,   "Error"   },
};

// @todo implementation: add logging to file
// @todo implementation: add ability to separate logging files for threads

template<class... Args>
void _app_log_message(LogType type, const char* tag, const char* file, int line, Args... args) {

    static const size_t bufSize = _MAX_PATH;
    static const size_t fileStrMinLen = 30;

    char buf[bufSize];
    auto msgSize = std::snprintf(buf, bufSize, args...);

    std::string fileStr = std::vformat("({}:{})", std::make_format_args(
                                           std::filesystem::path(file).filename().string(),
                                           line));
    size_t spacesCount = (fileStr.length() < fileStrMinLen)
                         ? fileStrMinLen - fileStr.length()
                         : 1u;
    std::string spaces(spacesCount, ' ');

    std::string output =
        std::vformat("{}  {}{}{}{}\t: {}{}",
                     std::make_format_args(tag, fileStr, spaces, logsTypeColors[type],
                         logsTypeStrings[type], std::string_view(buf, buf + msgSize), COLOR_RESET));

    std::printf("%s\n", output.c_str());
}

#else
#error "logs are not implemented for this platform"
#endif

#define PRINT_TAG_E(tag, ...) _app_log_message(LogType::Error,   tag, __FILE__, __LINE__, __VA_ARGS__)
#define PRINT_TAG_W(tag, ...) _app_log_message(LogType::Warning, tag, __FILE__, __LINE__, __VA_ARGS__)
#define PRINT_TAG_I(tag, ...) _app_log_message(LogType::Info,    tag, __FILE__, __LINE__, __VA_ARGS__)
#if DEBUG_LOGS
#define PRINT_TAG_V(tag, ...) _app_log_message(LogType::Debug,   tag, __FILE__, __LINE__, __VA_ARGS__)
#else
#define PRINT_TAG_V(tag, ...) do {} while(false)
#endif

#define PRINT_E(...) PRINT_TAG_E(LOGS_DEFAULT_TAG, __VA_ARGS__)
#define PRINT_W(...) PRINT_TAG_W(LOGS_DEFAULT_TAG, __VA_ARGS__)
#define PRINT_V(...) PRINT_TAG_V(LOGS_DEFAULT_TAG, __VA_ARGS__)
#define PRINT_I(...) PRINT_TAG_I(LOGS_DEFAULT_TAG, __VA_ARGS__)

#define PRINT(...) PRINT_I(__VA_ARGS__)
