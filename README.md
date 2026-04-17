# deep-learning
This repository implements a small deep learning framework written in C++ 23 with readability and simplicity in mind; it implements an automatic differentiation engine, computing kernels, a col-major nth-order tensor data structure, and a custom allocator to boost performance. It runs on the CPU and contains no architecture specific optimizations: it was written to learn about deep learning.

# Features
Training with mini batches is supported. Where possible, implementations of the functions that don't make use of the automatic differentiation engine (i.e. that don't allocate nodes) have been written to speed up inference.

Activation functions implemented ( ``src/TensorDeepLearning/Functions/ActivationFunctions.h`` ): Sin, ReLU, Softmax

Layer types implemented (``src/TensorDeepLearning/Layers`` ): Linear, Activation

Loss functions implemented ( ``src/TensorDeepLearning/Functions/LossFunctions.h`` ): Squared error, Cross-Entropy

Optimizers implemented (``src/TensorDeepLearning/Optimizers``): Stochastic Gradient Descent (SGD), AdaGrad, Adam, Adam with decoupled weight decay (AdamWD)

Learning rate schedulers implemented ( ``src/TensorDeepLearning/Schedulers`` ): Fixed rate

Implemented a model class to compose models of layers, retrieve optimization parameters, perform inference and write them to disk.

## Kernels
All kernels were written to be easy to understand first and foremost, and make use of operator overloading, they can be found in ``src/Tensor/TensorKernels.h``.

## Automatic differentiation
The automatic differentiation engine is the backbone of the framework, backpropagation is implemented recursively in the ``src/TensorAutomaticDifferentiation/TensorNodes.h``.

Nodes form the core of the engine, with every node taking care of producing its own gradient by taking the upstream gradient and applying the chain rule.

## Arena allocator
During development, it was observed that performing many fragmented heap allocations in a small time frame would lower performance, to allow the framework to use such allocation strategies, be easy to understand, yet fast, the placement new was used alongside a simple custom arena allocator; such allocator is implemented in the ``src/Arena`` folder, alongside some other utilities.

## Tensors
Tensors are implemented using col-major storage, the data structure's implementation can be found in ``src/Tensor/Tensor.h``.

## Tests
Tests are available for: the tensor class, which indirectly tests the arena; the automatic differentiation engine; the deep learning framework in a more practical manner.

Specifically, the deep learning framework has been tested by training and validating the results of a sample model.

## Examples
The deep learning framework can be seen in action in ``src/TensorDeepLearning/Tests/Classification.cpp`` where a simple neural network (composed of a 728x16 linear layer -> sin activation -> 16x10 linear layer) is trained to perform handwritten digit recognition using the MNIST handwritten digits dataset.

It is able to train for about 10 epochs in 30 seconds on a reasonably modern x86 CPU and ends up with an accuracy of about 91%.

Paths to the training and test sets should be provided on the CLI, the example project will take care of loading the dataset on its own.

## Notes
Meta programming is used heavily throughout the project to keep code repetition to a minimum when implementing element-wise operations; where possible transpositions are done by swapping indices instead of rewriting matrices in memory.
