# List of tasks to do


- ### Implementation

    The most important task for now. For now the project could open a given _.ptx_ and parse it.
    Even now parcing engine don't fully cover the _PTX_ spec. Such places are marked in code with `@todo implementation:`.

    Also there is a list of core features need to be implemented.

    The nearest tasks:

    1. Add ability to run kernels

        The kernels should be runned with a special cnd commnd. The data provided to kernal as an argument should be stored in storage in the format such as _json/xml/csv_ (i think _json_).

        So, the first task is to prepare CLI interface to run a kernel. Even in a single thread mode.

    1. The second is to implement an interpritator engine, executing the instructions from the given with using of the given data.

    1. The next task is to make a logic for multi-threaded kernel execution. The input data should be correctly shared between the threads.

    1. Current main goal is to support a correct execution of the "Add operation" kernel. It's source PTX is located in the [rel_add_op.ptx](.\ext\cuda_ptx_samples\rel_add_op.ptx) file.

    Long term tasks:

    1. Add a filtratio logic accorting to the _PTX_ version and gpu's _SM\_##_ type.

    1. Implement a threads' synchronization logic.

    1. Support more comlex kernels.


- ### Refactoring

    There are losts of places in the the code, that are marked with `@todo refactoring:`. This issues should be fixes to make code not looks like shit even for a little.


- ### Optimization

    Tryna not to worry about this for now. But the code in future need to be optimized. It seems to me that it is very inefficient now. A big amount of time should be made to improve the project efficiency.


- ### Documentation

    A documentation for the architecture and usage should be provided. The exciting small aount of docs should be updated before the code is publicly opened.