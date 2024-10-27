#pragma once

#include "tests.h"


namespace TestCase {

namespace Runtime {

struct RelAddOp final : public ITestCase {
    std::string     Name()        override;
    std::string     Description() override;
    PTX4CPU::Result Run()         override;
};

}  // namespace Runtime

}  // namespace TestCase
