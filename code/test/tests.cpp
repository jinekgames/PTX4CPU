#include "tests.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "rel_add_op.h"


using TestList = std::vector<const TestCase::ITestCase*>;


namespace {

TestList g_TestList = {
    &TestCase::Runtime::test_RelAddOp,
};

void CleanUp() { g_TestList.clear(); }

inline void PrintLineDelim() {
    std::cout << "============================================" << std::endl;
}

}  // anonimous namespace


static std::string ModifyDescription(const std::string& description) {

    auto ret = description;

    auto iter = ret.begin();
    for (;;) {
        ret.insert(iter, '\t');

        iter = std::find(iter, ret.end(), '\n');

        if(iter == ret.end()) {
            break;
        }
        ++iter;
    }

    return ret;
}


int main(int argc, char** argv) {

    std::cout << "Running automatic tests for PTX4CPU library" << std::endl;

    for (auto &pTest : g_TestList) {

        std::cout << std::endl;
        PrintLineDelim();
        std::cout << std::endl;

        std::cout << "Running test: " << pTest->Name()
                  << std::endl << std::endl;

        std::cout << "Description: " << std::endl
                  << std::endl << ModifyDescription(pTest->Description())
                  << std::endl << std::endl << std::endl;

        const auto res = pTest->Run();

        std::cout << std::endl;
        if (!res) {
            std::cout << "Test Failed. Error: "
                      << res.msg << std::endl;
        } else {
            std::cout << "Test Passed" << std::endl;
        }
    }

    CleanUp();

    return 0;
}
