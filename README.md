# Emulator which lets you run CUDA code on x86 CPU

**Current project state**: prototype enhancement.

**Project TODO:** [TODO.md](./TODO.md)


## What it will do?

Core functionality of the project is running of the CUDA PTX assemble code on the x86-based systems.
Using this you'll be able to run applications made for CUDA on platforms without CUDA support (platforms which do not include Nvidia GPU on board)

## How it works

The solution is to get compiled PTX code (which is generated by the _nvcc_) and emulate it to be ran on the CPU. The emulation consists of parsing of the PTX code and interpritating it on the runtime.

## Help

For cloning and building instruction look [here](./docs/building/build_instruction.md)

For external API usage instruction look [here](./code/core/emulator/api/export/README.md)

For command-line tool usage instruction look [here](./code/core/interfaces/emulator_cmd/README.md)

For architecture documentation look [here](./docs/core_functionality/architecture.md)

<br>

---
***2024, Kalinin Eugene***
