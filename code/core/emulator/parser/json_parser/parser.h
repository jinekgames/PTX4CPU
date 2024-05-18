#pragma once

#include <nlohmann/json.hpp>

#include <logger/logger.h>
#include <utils/result.h>
#include <utils/string_utils.h>
#include <parser_types.h>


namespace PTX4CPU {

namespace {

constexpr size_t JSON_VER_ELEMS_COUNT = 2;
using JsonVer = std::array<int, JSON_VER_ELEMS_COUNT>;

inline static JsonVer ParseJsonVer(const std::string& verStr) {

    JsonVer ver;

    auto strSplit = Split(verStr, '.');
    if (strSplit.size() != JSON_VER_ELEMS_COUNT)
        return {};

    for (auto i = 0; i < JSON_VER_ELEMS_COUNT; ++i) {
        ver[i] = std::stoi(strSplit[i]);
    }

    return ver;
}

template<Types::PTXType type>
void InsertScalarVar(PtxInputData& inputData, nlohmann::json& valueParser) {

    using RealType         = Types::getVarType<type>;
    constexpr auto ptrType = Types::GetSystemPtrType();
    using PtrRealType      = Types::getVarType<ptrType>;

    // The original value is passed to PTX code by it's address
    // We need to store both the orig value and it's address among the execution
    // time to not get out-of-date pointers.
    // The temp vars would not be used directly, they are only needed to keep
    // variables alive.

    // Init var value
    auto value = valueParser.get<RealType>();
    // Value converted to PTX variable
    Types::PTXVarPtr pPTXVar{new Types::PTXVarTyped<type>(&value)};
    // Retrive address of converted variable
    decltype(auto) pValTemp = &pPTXVar->Get<type>();
    // Save address of converted variable into the arg variable
    Types::PTXVarPtr pArgVar{new Types::PTXVarTyped<ptrType>(reinterpret_cast<PtrRealType*>(&pValTemp))};
    // Move converted vars into the data object
    inputData.tempVars.push_back(std::move(pPTXVar));
    inputData.execArgs.push_back(std::move(pArgVar));
}

template<Types::PTXType type>
void InsertVectorVar(PtxInputData& inputData, nlohmann::json& vectorParser) {

    using RealType         = Types::getVarType<type>;
    constexpr auto ptrType = Types::GetSystemPtrType();
    using PtrRealType      = Types::getVarType<ptrType>;

    Types::IndexType vectorSize = vectorParser.size();
    Types::PTXVarPtr pPTXVec{new Types::PTXVarTyped<type>(vectorSize)};

    for (Types::IndexType i = 0; i < vectorSize; ++i) {
        auto& valueParser = vectorParser[i];
        pPTXVec->Get<type>(i) = valueParser.get<RealType>();
    }
    // Retrive address of the 1st element of converted vector variable
    decltype(auto) pValTemp = &pPTXVec->Get<type>();
    // Save address of the 1st element of converted vector
    Types::PTXVarPtr pPTXpVar{new Types::PTXVarTyped<ptrType>(reinterpret_cast<PtrRealType*>(&pValTemp))};
    // Retrive address of the 1st element's address
    decltype(auto) ppValTemp = &pPTXpVar->Get<ptrType>();
    // Save address of the 1st element's address as an argument
    Types::PTXVarPtr pArgVar{new Types::PTXVarTyped<ptrType>(reinterpret_cast<PtrRealType*>(&ppValTemp))};
    // Move converted vars into the data object
    inputData.tempVars.push_back(std::move(pPTXVec));
    inputData.tempVars.push_back(std::move(pPTXpVar));
    inputData.execArgs.push_back(std::move(pArgVar));
}

} // anonimous namespace


/**
* Parces a given json configuring a PTX execution arguments
*
* @param jsonStr   a .json with execution arguments
* @param inputData object where PTX execution arguments and temporary
* variables will be put
*
* @return Parsing result
*/
inline static Result ParseJson(PtxInputData& inputData,
                               const std::string& jsonStr) {

    constexpr auto VER_KEY       = "version";
    constexpr auto ARGS_KEY      = "arguments";
    constexpr auto TYPE_KEY      = "type";
    constexpr auto VALUE_KEY     = "value";
    constexpr auto VECTOR_KEY    = "vector";

    inputData = PtxInputData{};

    try {

        auto thrw = [](std::string msg) {
            throw std::runtime_error(std::move(msg));
        };

        using namespace nlohmann;

        // Read json
        auto parser = json::parse(jsonStr);

        // Check version
        {
            auto verParser = parser[VER_KEY];
            auto verStr = verParser.get<std::string>();
            auto ver = ParseJsonVer(verStr);
            if (ver != JsonVer{1, 0}) {
                thrw("Invalid json ver. Must be 1.0. Got " + verStr);
            }
        }

        // Read args
        auto argsParser = parser[ARGS_KEY];
        for (const auto& argParser : argsParser) {

            if (!argParser.contains(TYPE_KEY)) {
                thrw("Variable must have 'type' field");
            }
            auto typeStr  = "." + argParser[TYPE_KEY].get<std::string>();
            bool isScalar = argParser.contains(VALUE_KEY);
            bool isVector = argParser.contains(VECTOR_KEY);
            if ((!isScalar && !isVector) || (isScalar && isVector)) {
                thrw("Variable must eigther have 'value' (for scalar) or "
                     "'vector' (for vector) field");
            }

            PRINT_V("Parsing %s json arg of type %s", (isScalar) ? "scalar" : "vector", typeStr.c_str());

            const auto type = Types::GetFromStr(typeStr);

            if (isScalar) {
                auto valueParser = argParser[VALUE_KEY];
                PTXTypedOp(type,
                    InsertScalarVar<_PtxType_>(inputData, valueParser);
                )
            } else {
                auto vectorParser = argParser[VECTOR_KEY];
                if (!vectorParser.is_array()) {
                    thrw("Vector object should be an array");
                }
                PTXTypedOp(type,
                    InsertVectorVar<_PtxType_>(inputData, vectorParser);
                )
            }
        }

    } catch (std::exception e) {
        return std::string("ERROR: Failed to parse arguments json. ") + e.what();
    }

    return {};
}

}  // namespace PTX4CPU
