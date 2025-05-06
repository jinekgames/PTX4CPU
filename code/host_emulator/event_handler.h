#pragma once

#include "emulator_api.h"
#include "utils/base_types.h"
#include "utils/api_types.h"

class EventHandler {
private:
    PTX4CPU::IEmulator* emulator_core_ = nullptr;

    PTX4CPU::PtxExecArgs args_;
    CudaTypes::uint3 grid_size_;

    std::string ptx_assembly_;
    std::string kernel_name_;

public:
    EventHandler() = default;
    EventHandler(EventHandler&&) = delete;
    EventHandler(EventHandler&) = delete;
    EventHandler& operator=(const EventHandler&) = delete;
    EventHandler& operator=(EventHandler&&) = delete;
    ~EventHandler() = default;

public:
    void LoadPtx();
    void SetArgs(void** args);
    void SetKernelName(const std::string name);
    void SetGridSize(const CudaTypes::uint3& grid_size);

    void EmuKernelLaunch() const;

    void InitEmulatorCore();
};
