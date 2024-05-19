#pragma once

#include <cstdint>
#include <string>


namespace PTX4CPU {

struct Result {

    enum class Code : uint32_t {
        Ok = 0,
        NotOk,
        Fail,
    };

    Result() : code(Code::Ok) {};
    Result(Code _code) : code(_code) {};
    Result(Code _code, std::string _msg)
        : code(_code)
        , msg(_msg) {};
    Result(std::string _msg)
        : code(Code::Fail)
        , msg(_msg) {};

    operator bool () const {
        return (code == Code::Ok);
    }

    bool operator == (Code _code) const {
        return (code == _code);
    }

    Code code = Code::Fail;
    std::string msg;
};

};  // namespace PTX4CPU
