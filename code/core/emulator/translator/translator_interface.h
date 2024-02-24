#pragma once

#include <string>


namespace PTX2ASM {

struct ITranslator {

    /**
     * Execute named function from the loaded PTX
     * 
     * @param funcName  compiled name of function
     * @param ...       execution args
     * 
     * @return void
    */
    virtual void ExecuteFunc(const std::string& funcName, ...) = 0;

};

};  // namespace PTX2ASM
