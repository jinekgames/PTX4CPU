#pragma once


#include <logger/logger.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#ifdef WIN32
#include <Windows.h>
#endif


namespace PTX4CPU {
namespace Helpers {


inline auto GetTick() {
    const auto time = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

inline void Sleep(uint64_t timeMs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(timeMs));
}

inline uint32_t GetLogicalThreadsCount() {
    return std::thread::hardware_concurrency();
}

class Timer final {

public:

    Timer(const std::string& name = "Unnamed")
        : m_Name(name)
        , m_StartTime(GetTick()) {}

    Timer(const Timer&) = delete;
    Timer(Timer&&)      = delete;

    ~Timer() {
        if (!m_IsFinished)
            Finish();
    }

    Timer& operator = (const Timer&) = delete;
    Timer& operator = (Timer&&)      = delete;

    bool IsFinished() { return m_IsFinished; }

    void Finish() {
        PRINT_I("%s timer: %llu ms", m_Name.c_str(), GetResult());
        m_IsFinished = true;
    }

    uint64_t GetResult() {
        return GetTick() - m_StartTime;
    }

private:

    bool m_IsFinished = false;
    uint64_t m_StartTime;
    std::string m_Name;

};


}  // namespace Helpers
}  // namespace PTX4CPU
