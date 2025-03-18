#include <iostream>
#include <getopt.h>

void print_help() {
    std::cout << "program [OPTIONS]\n"
              << "Options:\n"
              << "  -i, --input <file>   Input executable file\n";
}

int process(const std::string file) {
    auto cmd = std::string("LD_PRELOAD=libemulator_host.so EMU_OBJ_PATH=");
    cmd += file;
    cmd += " ";
    cmd += file;
    return system(cmd.c_str());
}

int main(int argc, char* argv[]) {
    static struct option long_options[] = {
        {"input",  required_argument, 0, 'i'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    std::string input_file;

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'h':
                print_help();
                return 0;
            default:
                print_help();
                return 1;
        }
    }

    if (input_file.empty()) {
        print_help();
        return 1;
    }

    return process(input_file);
}