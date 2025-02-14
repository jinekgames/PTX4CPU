#pragma once

#include <emulator_api.h>

#include <string>


namespace TestCase {

struct ITestCase {
    virtual std::string     Name()        = 0;
    virtual std::string     Description() = 0;
    virtual PTX4CPU::Result Run()         = 0;
};

}  // namespace TestCase
