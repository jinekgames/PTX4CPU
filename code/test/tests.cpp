#include "tests.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "rel_add_op.h"
#include "rel_add_op_const.h"


#ifndef DEFAULT_TESTS_ASSET_DIR
#error "Empty tests directory compile-time constant"
#endif


using TestList = std::vector<const TestCase::ITestCase*>;


namespace {

TestList g_TestList = {
    // &TestCase::Runtime::test_RelAddOp,
    &TestCase::Runtime::test_RelAddOpConst,
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

static std::string GetAssetPath(int argc, char** argv) {

    if (argc > 1) {
        return std::string{argv[1]};
    }

    return DEFAULT_TESTS_ASSET_DIR;
}


int main(int argc, char** argv) {

    std::map<std::string, std::string> fails;

    std::cout << "Running automatic tests for PTX4CPU library" << std::endl;

    const auto assetPath = GetAssetPath(argc, argv);
    std::cout << "Assets directory: " << assetPath.c_str() << std::endl;

    for (auto &pTest : g_TestList) {

        std::cout << std::endl;
        PrintLineDelim();
        std::cout << std::endl;

        std::cout << "Running test: " << pTest->Name()
                  << std::endl << std::endl;

        std::cout << "Description: " << std::endl
                  << std::endl << ModifyDescription(pTest->Description())
                  << std::endl << std::endl << std::endl;

        std::cout << "0, " << assetPath.c_str() << "test: " << (void*)pTest
                  << "test func: " << (void*)(&pTest->Run) << std::endl;

        const auto res = pTest->Run(assetPath);

        std::cout << std::endl;
        if (!res) {
            fails[pTest->Name()] = res.msg;
            std::cout << "Test Failed. Error: "
                      << res.msg << std::endl;
        } else {
            std::cout << "Test Passed" << std::endl;
        }
    }

    CleanUp();

    if (fails.empty()) {
        return 0;
    }

    std::cout << std::endl
              << "Test failed. "
              << fails.size() << " test cases failed:" << std::endl;

    for (const auto& el : fails) {
        std::cout << " - " << el.first << " : " << el.second <<  std::endl;
    }

    std::cout << std::endl;

    return 1;
}
