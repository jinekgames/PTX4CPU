# Configuring of kernel execution arguments with `.json`

Current supported `.json` version: `1.0`

## Example

[Here](../../../../../ext/cuda_ptx_samples/rel_add_op.json) you can see the example of a valid `.json`. Near the file you'll see a PTX with the corespondent name, for which the `.json` was created.

## `.json` structure

The typical `.json` syntaxis is used.

In a lower level you need to specify 2 fields:

- `version (str)` with configuration file version
- `arguments (array)` with the arguments themself

### `arguments` field configuring

The field `arguments` must be an array of arguments.

Arguments are setted in the fasion they were passed to the kernel in the `.cu` high-level code. It need to be stated, that arguments passed in the HOST code, are provided by value (reference), when the PTX code gets the memory addresses of these arguents. So, typically all of types in the PTX are `u64` (or `u32` if you are a kind of a time traveler), but types in the HOST code could be any of allowed. It is very necessary to setup types correctly according to the HOST-side high-level code.

When arguments from `.json` are being parsed, they are converted into the addresses of the values you put into the `.json` to be processed on execution.

An argument is an object with 2 fields:

- `type (str)` - a PTX type of variable. Standard PTX types must be used (e.g. `s32`, `f64`, `b8`) (but it's actually a HOST-side _C++_ type)
- `value (numerical)` or `vector (array)` with the value of argument

#### Configuring argument value

There are 2 options how you can set up a value depending on the argument representation. Depending on scalarity of variable, eigther a `value` or `vector` field need to be used. Using both of them at once is prohibited.

If the argument is a simple scalar value, the `value` field is the appropriate one. The value of a `value` field should be a number with the type corresponding to the `type` field.

If your variable is an array of values, the `vector` field is your one.
If you want to specify initial values of an array, the value of `vector` field must be an array of numerical values.
If you want skip array's content initialization (e.g. this array is used as a result variable), just set the size of the array in `vector` field's value.
Then address of the 1st element of the array will be used.
