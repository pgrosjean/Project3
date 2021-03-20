# Project 3 - Neural Networks
## Due 03/19/2021

![BuildStatus](https://github.com/pgrosjean/Project3/workflows/HW3/badge.svg?event=push)

## This Repository
This repo holds all of the information needed to generate and run a neural network for both a binary classification task as well as an autoencoding class.
This repository is set up with three main sections locations for the (1) Docs (2) Scripts Module and submodules and (3) the pytest functionalities. The scripts modules is broken into 4 submodules:

NN: Neural Network related classes/functions
io: Import export related classes/functions
preprocess: Preprocessing realted classes/functions
optim: Particle Swarn Optimization related classes/functions

### Docs
To view the API docs for the align module and specifically the classes contained within scripts and the 4 submodules please see the [API doc](https://github.com/pgrosjean/Project3/blob/main/docs/build/index.html) by running "open index.html" in the location of the file pointed to by the API doc link.

### Scripts Module
The [Scripts Module](https://github.com/pgrosjean/Project3/tree/main/scripts) holds the four submodules described above that are used in the write up for this project that can be found [here](https://github.com/pgrosjean/Project3/blob/main/Parker_Grosjean_Project3.ipynb). These submodules contain classes and functions that allow a user to generate and train neural networks with a variety of architectures. Please refer to the docs above to see the module contents and how they should be used.

### Pytest Functionalities
All of the neccessary unit tests are implemented [here](https://github.com/pgrosjean/Project3/blob/main/test/test_NN.py).

### testing
Testing is as simple as running
```
python -m pytest test/*
```
from the root directory of this project.
