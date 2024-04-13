#include <logger.h>
#include <translator_interface.h>
#include <version.h>


using namespace PTX4CPU;

ITranslator::ITranslator() {
    PRINT_I("Initializing a PTX Translator (Git commit %s)", ProjectGitCommit);
}
