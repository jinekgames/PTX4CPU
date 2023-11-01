#pragma once

#include <string>


namespace PTX2ASM {

struct ITranslator {

    /**
     * Translate loaded PTX to ASM code
     * 
     * @returns true if suceeded; false, if translation failed
    */
    virtual bool Translate() = 0;
    /**
     * Returns translated code string
     * 
     * @returns string with translated code if PTX has been loaded and translated, empty string in the other case
    */
    virtual std::string GetResult() const = 0;

};

};  // namespace PTX2ASM
