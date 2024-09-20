# List of tasks to do


- ### Implementation

    The most important task for now. For now the project could open a given _.ptx_ and parse it. It can also parse and half-run an 'add' PTX sample.
    Even now parcing engine don't fully cover the _PTX_ spec (if not talking about the all operations' support). Such places are marked in code with `@todo implementation:`.

    We currently can run a PTX with the command line. The PTX which support is currently implemented is located in the [rel_add_op.ptx](.\ext\cuda_ptx_samples\rel_add_op.ptx) file.

    Also there is a list of core features need to be implemented.

    The nearest tasks:

    1. Parsing engine redesign

        Current implementation of PTX-file parser is very poor and won't allow to cover the PTX language features. It is to be redesigned the way the instruction is parsed and stored. May be it will be needed to refactor the ibstructions torage format too, but currently not planned

    1. Add autotests (locally-runned)

        It is about creating the environment for intergaration autotests to be runned locally. They should run the supported PTX samples and check the output `.json` with the expected.

        It is also a good idea to make some unittests environment.

    Long term tasks:

    1. Add a filtration logic accorting to the _PTX_ version and gpu's _SM\_##_ type.

    1. Implement a threads' synchronization logic.

    1. Support more comlex kernels (_there should be the list_). It is about coverage of PTX. The following things are now planned to be supported: predicates, synchronizations, C-style dirictives, includes.

    1. Provide warp-based execution, blocked execution, shared memory.

    1. Provide efficient multithreaded execution (see the **Optimization** paragraph).


- ### Refactoring

    There are losts of places in the the code, that are marked with `@todo refactoring:`. This issues should be fixes to make code not looks like shit even for a little.


- ### Optimization

    Tryna not to worry about this for now. But the code in future need to be optimized. It seems to me that it is very inefficient now. A big amount of time should be made to improve the project efficiency.

    I have the following ideas about the architecture improvement, which will make the efficiency better:

    - A kernel is ran on as many threads is possible currently. It'll be better to create only as many logical threads, as the system has physiscally. The threads should be executed till the some break (e.g. synchronization point) and the logical thread should switch the execution to another ready thread (not stated in another logical thread to try keep l1 and l2 caches)

    - We can improve memory consuption by simplifying the used data types.

    - I struction's parsing logic should be removed from the threads executions tep into the prepass, which will prepare the low-memory list of isntructions and links to the arguments


- ### Documentation

    A documentation for the architecture and usage should be provided. The existed small amount of docs should be updated before the code is publicly opened.