#pragma once

#include <string>
#include <unordered_map>

class Parser : public std::unordered_map<std::string, std::string> {

public:
    Parser(size_t argc, char** argv) {
        Parse(argc, argv);
    }
    Parser(const Parser&) = delete;
    Parser(Parser&&) = delete;
    Parser operator = (const Parser&) = delete;
    Parser operator = (Parser&&) = delete;
    ~Parser() = default;

    bool Contains(const std::string& key) {
        return find(key) != end();
    }

private:
    void Parse(size_t argc, char** argv);

};
