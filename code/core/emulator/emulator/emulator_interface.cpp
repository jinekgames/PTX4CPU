#include <logger/logger.h>
#include <emulator/emulator_interface.h>
#include <utils/string_utils.h>
#include <version.h>


using namespace PTX4CPU;

IEmulator::IEmulator() {
    PRINT_I("Initializing a PTX Translator v%s (Git commit %s)",
            kProductVersion, kProjectGitCommit);
}
