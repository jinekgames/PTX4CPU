# Usage of tool

Tool has a comand-line interface.

It has 2 base functionalities:

- test initialization of a PTX file;
- executing of a kernel from a PTX file.

Command-line arguments can be seen by running the tool with `--help` argument.

## Test initialization of PTX file

The only argument is a path to the PTX.

Setted file will be tryed to be loaded and preparsed.

It is rather a debug functionality, than a useful feature.

## Executing of a kernel from a PTX file

You will be asked to set:

- path to the PTX
- name of kernel to execute
- count of execution threads
- execution arguments provided via a `.json` file

This command will fully init the PTX and run the kernel in a multithreaded fashion using only the CPU power.

It is the core feature you are looking for if you are reading this doc.

User should manually prepare a configuration `.json` file with arguments for the kernel. Instruction on how to do this available [here](../../emulator/parser/arg_parsers/json_parser/README.md).
