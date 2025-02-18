#pragma once

#include <emulator_api.h>

#include <string>


namespace TestCase {

struct ITestCase {
    virtual std::string     Name()                                const = 0;
    virtual std::string     Description()                         const = 0;
    virtual PTX4CPU::Result Run(const std::string& testAssetPath) const = 0;

    virtual ~ITestCase() = default;

    ITestCase()                 = default;
    ITestCase(const ITestCase&) = default;
    ITestCase(ITestCase&&)      = default;
    ITestCase& operator = (const ITestCase&) = default;
    ITestCase& operator = (ITestCase&&)      = default;
};

namespace Runtime {

inline constexpr auto kNamePrefix = "[Runtime] ";

}  // namespace Runtime

}  // namespace TestCase
