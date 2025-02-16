#pragma once

#ifdef OPT_COMPILE_SAFE_CHECKS
#ifdef WIN32
namespace Win32
{
#include <Windows.h>
}  // namespace Win32
#endif  // #ifdef WIN32
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS


#ifdef OPT_COMPILE_SAFE_CHECKS

namespace Validator {

template<class Type>
void CheckPointer(const Type* ptr) {
#ifdef OPT_COMPILE_SAFE_CHECKS
    if (!ptr) {
        PRINT_E("Null pointer dereference");
        return;
    }
#ifdef WIN32
    if (Win32::IsBadReadPtr(reinterpret_cast<void*>(
                                const_cast<Type*>(ptr)),
                            static_cast<Win32::UINT_PTR>(sizeof(Type)))) {
        PRINT_E("Invalid pointer. Possible reading access violation");
        return;
    }
#else  // #ifdef WIN32
#pragma message("Pointer validation is not supported on your platform")
#endif  // #ifdef WIN32
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS
}

}  // namespace Validator

#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS
