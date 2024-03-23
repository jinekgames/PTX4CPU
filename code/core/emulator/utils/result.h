#pragma once

#include <cstdint>
#include <string>


namespace PTX2ASM {

enum class ResultCode : uint32_t {
    Ok = 0,
    Fail,
};

struct Result {

    Result() : code(ResultCode::Ok) {};
    Result(ResultCode _code) : code(_code) {};
    Result(std::string _msg)
        : code(ResultCode::Fail)
        , msg(_msg) {};

    operator bool () {
        return (code == ResultCode::Ok);
    }

    ResultCode code = ResultCode::Fail;
    std::string msg;
};

};  // namespace PTX2ASM
