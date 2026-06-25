# reference

this folder contains reference implementation of a specific op. Note by including a specific header, you are including the implementation(expecially the gpu implementation) into your source code, and compile that kernel into the fatbin, hence may increase your kernel obj code length. Usually the header starts with `reference_` is a cpu reference implementation. The header starts with `naive_` contains a gpu implementation with a small launcher.

TODO: move `host/reference` under this folder
