# Emulator API overview

The main part of the _Emulator_ is a library (static or dynamic) which implements the backend of a _PTX_ emulation process. The library provides an API for using it's power of _PTX_ emulation.

The API interface consists of the following header files:

- The core header file, which includes other API headers, and provides the basic interface of working with the library:
  - [`emulator_api.h`](./emulator_api.h)
- The emulator base class definition, which provides the core emulation features:
  - [`emulator/emulator_interface.h`](./emulator/emulator_interface.h)
- Some misc headers defining library helper types:
  - [`utils/result.h`](./utils/result.h)
  - [`utils/base_types.h`](./utils/base_types.h)
  - [`api_types.h`](./api_types.h)

All these headers are stored in directory `code/core/emulator/api/export`. After building they are copied to


## API usage

The core functionality of the library is an ability to load and launch a kernel from a PTX file. This section describes how it could be done using the API step by step. Each mentioned function is linked with the file defining it. Detailed documentation for each function is available in the code.

1. First of all we need to create an `Emulator` object, which will be used for kernel executing. `Emulator` object stores the _PTX_ file and provides methods for _PTX_ execution and overviewing.

   To create an `Emulator` object you should call [`EMULATOR_CreateEmulator()`](./emulator_api.h). You will need to pass a _PTX_ files in text format to be loaded into the `Emulator`.

1. When the `Emulator` is successfully created we now need to prepare argumetsfor the kernel execution. Depending on the nature arguments, they could be eigher a `void**` pointer (passed when using the library from the CUDA Runtime) or a configuration [`json` file](../../emulator/parser/ext_parsers/json_parser/README.md) (when running a _PTX_ out of the Runtime). So, there are 2 correspondent ways of parsing arguments.

   - To parse Runtime `void**` args, you should use [`EMULATOR_ProcessArgs()`](./emulator_api.h). This function requires a kernel descriptor for non-typed args processing. To retrive a descriptor you should call an `Emulator` method [`GetKernelDescriptor()`](./emulator/emulator_interface.h) with the name of a kernel the arguments are passed for.

   - To parse arguments from a configuration _json_ file, call [`EMULATOR_ParseArgsJson()`](./emulator_api.h) with the content of a configuration file.

   Both of the ways will produce a descriptor for the kernel execution arguments, which is used for executing the kernel.

1. Now you are ready for executing a kernel. It could be done directly from an `Emulator` object by calling the method [`ExecuteFunc()`](./emulator/emulator_interface.h). THis will execute the specified kernel from a _PTX_ loaded into the `Emulator` in the specified threads count with the arguments specified with the desciptor you got on a previous step.

1. Receiving results.

   - If the arguments were passed from the Runtime, you need to do nothing, because the values are passed to the kernel via pointers. So, after the `Emulator` finished execution, all your application's data will already contain modified values.

   - If you passed arguments by a _json_, you can serialize modified values back with [`EMULATOR_SerializeArgsJson()`](./emulator_api.h).
