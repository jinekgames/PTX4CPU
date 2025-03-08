#pragma once

namespace PTX4CPU {
namespace Debug {

inline void UnconditionalBreakpoint() noexcept {

#if __cpp_lib_debugging >= 202311L

    if (std::is_debugger_present()) {
        std::breakpoint();
    }

#else  // __cpp_lib_debugging < 202311L

#if defined(WIN32)

    __debugbreak();

#elif defined(UNIX)

     __asm__ volatile("int $0x03");

#else

#pragma message("UnconditionalBreakpoint() is not supported on your platform")

#endif  // defined(WIN32) || defined(UNIX)

#endif  // __cpp_lib_debugging >= 202311L

}

}  // namespace Debug
}  // namespace PTX4CPU
