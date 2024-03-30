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
# train the model using sample data
$ ./demo.sh

# install libraries needed for plotting
$ pip install scikit-learn
$ pip install matplotlib
$ pip install seaborn

```