#pragma once

#include <nlohmann/json.hpp>

#include <logger/logger.h>
#include <utils/result.h>
#include <utils/string_utils.h>
#include <parser_types.h>

#include <sstream>


namespace PTX4CPU {

namespace {

constexpr size_t JSON_VER_ELEMS_COUNT = 2;

class JsonVer : public std::array<int, JSON_VER_ELEMS_COUNT> {
public:
    void Parse(const std::string& verStr) {
        auto strSplit = Split(verStr, '.');
        if (strSplit.size() != JSON_VER_ELEMS_COUNT)
            return;

        for (auto i = 0; i < JSON_VER_ELEMS_COUNT; ++i) {
            at(i) = std::stoi(strSplit[i]);
        }
    }
    std::string ToStr() const {
        std::stringstream ret;
        for (size_t i = 0; i < JSON_VER_ELEMS_COUNT; ++i) {
            ret << at(i);
            if (i < JSON_VER_ELEMS_COUNT - 1) {
                ret << ".";
            }
        }
        return ret.str();
    }
};

constexpr JsonVer JSON_VER_ACTUAL = { 1, 0 };

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
    std::stringstream ss{std::ios::in};
    // Value converted to PTX variable
    Types::PTXVarPtr pPTXVar{new Types::PTXVarTyped<type>(&value)};
    // Retrive address of converted variable
    decltype(auto) pValTemp = &pPTXVar->Get<type>();
    // Save address of converted variable into the arg variable
    Types::PTXVarPtr pArgVar{new Types::PTXVarTyped<ptrType>(reinterpret_cast<PtrRealType*>(&pValTemp))};
    // Move converted vars into the data object
    inputData.outVars.push_back(std::move(pPTXVar));
    inputData.execArgs.push_back(std::move(pArgVar));
}

template<Types::PTXType type>
void InsertVectorVar(PtxInputData& inputData, nlohmann::json& vectorParser) {

    using RealType         = Types::getVarType<type>;
    constexpr auto ptrType = Types::GetSystemPtrType();
    using PtrRealType      = Types::getVarType<ptrType>;

    Types::IndexType vectorSize = 0;
    bool writeVectorData = true;
    if (vectorParser.is_array()) {
        vectorSize = vectorParser.size();
    } else {
        vectorSize = vectorParser.get<Types::IndexType>();
        writeVectorData = false;
    }

    Types::PTXVarPtr pPTXVec{new Types::PTXVarTyped<type>(vectorSize)};
    if (writeVectorData) {
        for (Types::IndexType i = 0; i < vectorSize; ++i) {
            auto& valueParser = vectorParser[i];
            pPTXVec->Get<type>(i) = valueParser.get<RealType>();
        }
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
    inputData.outVars.push_back(std::move(pPTXVec));
    inputData.tempVars.push_back(std::move(pPTXpVar));
    inputData.execArgs.push_back(std::move(pArgVar));
}

template<Types::PTXType type>
void ExportScalarVar(nlohmann::json& valueParser, const PTX4CPU::Types::PTXVarPtr& pPTXVar) {

    valueParser = pPTXVar->Get<type>();
}

template<Types::PTXType type>
void ExportVectorVar(nlohmann::json& valueParser, const PTX4CPU::Types::PTXVarPtr& pPTXVar) {

    valueParser = nlohmann::json::array();
    for (PTX4CPU::Types::IndexType i = 0; i < pPTXVar->GetDynamicSize(); ++i) {
        const auto elementValue = pPTXVar->Get<type>(i);
        valueParser.push_back(elementValue);
    }
}


constexpr auto VER_KEY       = "version";
constexpr auto ARGS_KEY      = "arguments";
constexpr auto TYPE_KEY      = "type";
constexpr auto VALUE_KEY     = "value";
constexpr auto VECTOR_KEY    = "vector";

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

    inputData = PtxInputData{};

    try {

        auto thrw = [](std::string msg) {
            throw std::runtime_error(std::move(msg));
        };

        using namespace nlohmann;

        auto checkField = [&](const json& parser, const std::string& parentName,
                             std::string fieldName) {
            if (!parser.contains(fieldName)) {
                thrw(parentName + " must contain a `" + fieldName + "` field");
            }
        };

        // Read json
        auto parser = json::parse(jsonStr);

        // Check version
        {
            checkField(parser, "Json", VER_KEY);
            auto verParser = parser[VER_KEY];
            auto verStr = verParser.get<std::string>();
            JsonVer ver;
            ver.Parse(verStr);
            if (ver != JsonVer{1, 0}) {
                thrw("Invalid json ver. Must be 1.0. Got " + verStr);
            }
        }

        // Read args
        checkField(parser, "Json", VER_KEY);
        auto argsParser = parser[ARGS_KEY];
        for (const auto& argParser : argsParser) {

            checkField(argParser, "Variable", TYPE_KEY);
            auto typeStr  = "." + argParser[TYPE_KEY].get<std::string>();
            bool isScalar = argParser.contains(VALUE_KEY);
            bool isVector = argParser.contains(VECTOR_KEY);
            if ((!isScalar && !isVector) || (isScalar && isVector)) {
                thrw("Variable must eigther have 'value' (for scalar) or "
                     "'vector' (for vector) field");
            }

            PRINT_V("Parsing %s json arg of type %s", (isScalar) ? "scalar" : "vector", typeStr.c_str());

            const auto type = Types::StrToPTXType(typeStr);

            if (isScalar) {
                auto valueParser = argParser[VALUE_KEY];
                PTXTypedOp(type,
                    InsertScalarVar<_PtxType_>(inputData, valueParser);
                )
            } else {
                auto vectorParser = argParser[VECTOR_KEY];
                PTXTypedOp(type,
                    InsertVectorVar<_PtxType_>(inputData, vectorParser);
                )
            }
        }

    } catch (std::exception e) {
        return std::string("Failed to parse arguments json. Error: ") + e.what();
    }

    return {};
}


/**
 * Serializes the resulting PTX arguments' values into the json
 *
 * @param inputData object where PTX execution result and temporary
 * variables are stored
 * @param jsonStr   an output .json with execution resuts
 *
 * @return Serialization result
*/
inline static Result ExtractJson(const PtxInputData& inputData,
                                 std::string& jsonStr) {

    jsonStr.clear();

    try {

        auto thrw = [](std::string msg) {
            throw std::runtime_error(std::move(msg));
        };

        using namespace nlohmann;

        json parser;

        parser[VER_KEY]  = JSON_VER_ACTUAL.ToStr().c_str();

        auto& argsParser = parser[ARGS_KEY];
        argsParser = json::array();
        for (const auto& pPTXVar : inputData.outVars) {

            json argParser;

            const auto type = pPTXVar->GetPTXType();
            auto typeStr = Types::PTXTypeToStr(type);
            typeStr.erase(0, 1);
            argParser[TYPE_KEY] = typeStr.c_str();

            const bool isScalar = (pPTXVar->GetDynamicSize() == 1);
            if (isScalar) {
                auto& valueParser = argParser[VALUE_KEY];
                PTXTypedOp(type,
                    ExportScalarVar<_PtxType_>(valueParser, pPTXVar);
                )
            } else {
                auto& valueParser = argParser[VECTOR_KEY];
                PTXTypedOp(type,
                    ExportVectorVar<_PtxType_>(valueParser, pPTXVar);
                )
            }

            argsParser.push_back(argParser);
        }

        jsonStr = parser.dump(4, ' ', true);

    } catch (std::exception e) {
        return std::string("ERROR: Failed to parse arguments json. ") + e.what();
    }

    return {};
}

}  // namespace PTX4CPU
