#pragma once

#include <cstdint>
#include <string>


namespace PTX4CPU {

struct Result {

    enum class Code : uint32_t {
        Ok = 0,
        Fail,
    };

    Result() : code(Code::Ok) {};
    Result(Code _code) : code(_code) {};
    Result(std::string _msg)
        : code(Code::Fail)
        , msg(_msg) {};

    operator bool () const {
        return (code == Code::Ok);
    }

    Code code = Code::Fail;
    std::string msg;
};

};  // namespace PTX4CPU
