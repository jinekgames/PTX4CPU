#include <logger/logger.h>
#include <translator_interface.h>
#include <utils/string_utils.h>
#include <version.h>


using namespace PTX4CPU;

ITranslator::ITranslator() {
    PRINT_I("Initializing a PTX Translator v%s (Git commit %s)", PTX4CPU_VERSION, ProjectGitCommit);
}
