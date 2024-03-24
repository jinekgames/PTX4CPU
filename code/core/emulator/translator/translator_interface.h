#pragma once

#include <string>

#include <utils/result.h>


namespace PTX2ASM {

struct ITranslator {

    ITranslator();

    /**
     * Execute named function from the loaded PTX
     *
     * @param funcName  compiled name of function
     * @param ...       execution args
     *
     * @return Result
    */
    virtual Result ExecuteFunc(const std::string& funcName) = 0;

};

};  // namespace PTX2ASM
