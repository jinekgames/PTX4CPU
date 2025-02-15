#pragma once

#include "tests.h"


namespace TestCase {

namespace Runtime {

inline constexpr auto kNamePrefix = "[Runtime] ";

struct RelAddOp final : public ITestCase {

    static constexpr auto kName = "rel_add_op (vadd)";

    std::string     Name()                                const override;
    std::string     Description()                         const override;
    PTX4CPU::Result Run(const std::string& testAssetPath) const override;

    RelAddOp()                = default;
    RelAddOp(const RelAddOp&) = default;
    RelAddOp(RelAddOp&&)      = default;
    ~RelAddOp()               = default;
    RelAddOp& operator = (const RelAddOp&) = default;
    RelAddOp& operator = (RelAddOp&&)      = default;
};

inline const RelAddOp test_RelAddOp;

}  // namespace Runtime

}  // namespace TestCase
