#include <logger.h>
#include <translator_interface.h>
#include <version.h>


namespace PTX2ASM {

ITranslator::ITranslator() {
    PRINT_I("Initializing a PTX Translator (Git commit %s)", ProjectGitCommit);
}

};  // namespace PTX2ASM