# Pre-requisites
1. GNU Make
2. GCC (Clang pretending to be GCC is fine)


Initial configuration using conda
```bash
$ conda create -n glove python=3.8
$ conda activate glove
$ pip install numpy
# build executable
$ make
```