Currently is supported Windows x64 and Linux x64 only.

It was tried to write code in platform-independent fasion, so, it will not be a
big deal to support another build target (even the Android OS or
HarmonyOS NEXT :/)

**Clone**

```
git clone https://github.com/jinekgames/PTX4CPU.git
cd PTX4CPU
git submodule update --init --recursive
```

**Build**

For building you need a compiler with C++ 23 support (e.g. GCC 13.2.0) and
CMake.

```
mkdir build
cd build
cmake ..
cmake --build .
```

or just use any other ability to build from the CMakeLists.txt stored in the
project's root directory.
