#include <fstream>
#include <filesystem>

#include "event_handler.h"
#include "logger/logger.h"

namespace fs = std::filesystem;

void EventHandler::LoadPtx()
{
    auto object = fs::path(getenv("EMU_OBJ_PATH"));

    if(!fs::is_regular_file(object)) {
        PRINT_E("Invalid object path \t%s", object.c_str());
        return;
    }
    std::array<char, 512> buffer;
    std::string result;
    auto cmd = std::string("cuobjdump -ptx ");
    cmd += object;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if(pipe == nullptr) {
        PRINT_E("Decoding object error\t%s", object.c_str());
        return;
    }
    while(fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if(result.empty()) {
        PRINT_E("Ptx section is empty");
        return;
    }

    ptx_assembly_ = result;
}

void EventHandler::InitEmulatorCore()
{
    if(ptx_assembly_.empty()) {
        PRINT_E("Ptx is empty");
        return;
    }
    EMULATOR_CreateEmulator(&emulator_core_, ptx_assembly_);
}

void EventHandler::SetArgs(void** args)
{
    if(emulator_core_ == nullptr) {
        PRINT_E("Emulator core has not been initialized");
        return;
    }
    PTX4CPU::PtxFuncDescriptor desc_{};
    emulator_core_->GetKernelDescriptor(kernel_name_, &desc_);
    EMULATOR_CreateArgs(&args_, desc_, args);
    if(emulator_core_ == nullptr) {
        PRINT_E("Arguments has not been setted");
        return;
    }
}

void EventHandler::SetGridSize(const CudaTypes::uint3& grid_size)
{
    grid_size_ = grid_size;
}

void EventHandler::EmuKernelLaunch() const
{
    if(emulator_core_ == nullptr) {
        PRINT_E("Emulator core has not been initialized");
        return;
    }
    if(emulator_core_ == nullptr) {
        PRINT_E("Incorrect kernel name");
        return;
    }
    emulator_core_->ExecuteFunc(kernel_name_, args_, grid_size_);
}

void EventHandler::SetKernelName(const std::string name)
{
    kernel_name_ = name;
}
