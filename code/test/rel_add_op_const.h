#pragma once

#include "tests.h"


namespace TestCase {

namespace Runtime {

struct RelAddOpConst final : public ITestCase {

    static constexpr auto kName = "rel_add_op_const (vadd 1 argument)";

    std::string     Name()                                const override;
    std::string     Description()                         const override;
    PTX4CPU::Result Run(const std::string& testAssetPath) const override;

    RelAddOpConst()                     = default;
    RelAddOpConst(const RelAddOpConst&) = default;
    RelAddOpConst(RelAddOpConst&&)      = default;
    ~RelAddOpConst()                    = default;
    RelAddOpConst& operator = (const RelAddOpConst&) = default;
    RelAddOpConst& operator = (RelAddOpConst&&)      = default;
};

inline const RelAddOpConst test_RelAddOpConst;

}  // namespace Runtime

}  // namespace TestCase
