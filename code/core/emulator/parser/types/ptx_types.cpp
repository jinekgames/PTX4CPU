#include "ptx_types.h"

#include <algorithm>

#include <logger/logger.h>


#if !HAS_FIXED_FLOAT_SUPPORT
#pragma message("WARNING: "                                               \
                "Fixed floating point types are not supported. "          \
                "All floats with size > 64 will be traited as float64")
#endif  // #if !HAS_FIXED_FLOAT_SUPPORT


namespace PTX4CPU {
namespace Types {

namespace {

inline static const std::unordered_map<std::string, PTXType> StrTable = {
    { ".b8",    PTXType::B8    },
    { ".b16",   PTXType::B16   },
    { ".b32",   PTXType::B32   },
    { ".b64",   PTXType::B64   },
    // { ".b128",  PTXType::B128  }, // unsupported

    { ".s8",    PTXType::S8    },
    { ".s16",   PTXType::S16   },
    { ".s32",   PTXType::S32   },
    { ".s64",   PTXType::S64   },

    { ".u8",    PTXType::U8    },
    { ".u16",   PTXType::U16   },
    { ".u32",   PTXType::U32   },
    { ".u64",   PTXType::U64   },

    { ".f16",   PTXType::F16   },
    { ".f16x2", PTXType::F16X2 },
    { ".f32",   PTXType::F32   },
    { ".f64",   PTXType::F64   },

    // { ".pred",  PTXType::Pred  }, // unsupported
};

}  // anonimous namespace

PTXType StrToPTXType(const std::string& typeStr) {
    if (StrTable.contains(typeStr)) {
        return StrTable.at(typeStr);
    }
    PRINT_E("Unknown type of variable: \"%s\". Treating as .s64", typeStr.c_str());
    return Types::PTXType::S64;
}

std::string PTXTypeToStr(PTXType type) {
    const auto& found = std::find_if(StrTable.begin(), StrTable.end(),
        [&](const std::pair<std::string, PTXType>& el) {
            if (el.second == type)
                return true;
            return false;
        }
    );
    if (found != StrTable.end()) {
        return found->first;
    }
    PRINT_E("Unknown type of variable: PTXType(%lu)", static_cast<uint32_t>(type));
    return {};
}

}  // namespace Types
}  // namespace PTX4CPU
