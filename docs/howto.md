# Neural network in CUDA/C++

• <strong>[Description](#description)</strong><br>
• <strong>[Usage](#usage)</strong><br>

## Description
This is an implementation of a neural net, completely from scratch, in CUDA/C++.

## Usage
The code is by no means efficient and is meant as an introduction to CUDA only. Here is an overview of the various classes and functions:

Everything is implemented in both pure C++ (under ```CPU/```) and CUDA/C++ (under ```GPU/```). The syntax remains virtually identical, and there are only two points to bear in mind when switching between C++ and CUDA/C++:

1. C++ and CUDA/C++ modules end with the suffixes ```CPU``` and ```GPU``` respectively.
2. Don't forget to allocate and destroy CUDA arrays via ```cudaMallocManaged``` and ```cudaFree```

* ```linear.h/Linear_SUFFIX```: 
  * Initialization: 

    Required arguments: 
     * ```_bs``` (```int```): Batch size.
     * ```_n_in``` (```int```): Number of input features.
     * ```_n_out``` (```int```): Number of output features.
    
    Optional arguments:
     * ```_lr``` (```float```): Learning rate.
    
  * ```forward```: Runs a linear forward pass.

    Required arguments:
     * ```_inp``` (```float*```): Pointer to the input data.
     * ```_out``` (```float*```): Pointer for storing the output data.
    
  * ```update```: Updates the weights and biases.

  * ```backward```: Performs a backward pass, storing the gradients in ```_inp```.

* ```relu.h/ReLU_SUFFIX```:
  * Initialization:

    Required argument:
     * ```_sz_out``` (```int```): The number of input/output elements.
    
  * ```forward```, ```backward```: Like ```Linear_SUFFIX``` but for ReLU.

* ```mse.h/MSE_SUFFIX```:
  * Initialization: Like ReLU.

  * ```forward```: Dummy method for compatibility with the other modules and performing backpropagation; does not actually calculate the loss.

    Required arguments:
     * ```_inp``` (```float*```): Pointer to the predictions.
     * ```_out``` (```float*```): Pointer to the target values.
    
  * ```_forward```: Calculates the MSE. This method is solely for calculating the loss and cannot be used during backpropagation.

    Required arguments: Like ```MSE_SUFFIX``` but ```_out``` must have an extra element for storing the loss.
    
  * ```backward```: Performs a backward pass, storing the gradients in ```_inp```.

* ```sequential.h/Sequential_SUFFIX```:

  * Initialization: 

    Required arguments:
     * ```layers``` (```std::vector<Module*>```): Layers to be chained together.
    
  * ```forward```: Cascades the modules in ```layers```.

    Required arguments:
     * ```inp``` (```float*```): Pointer to the input data.
     * ```out``` (```float*```): Dummy argument, only for compatibility with the other forward methods and doesn't get used. The output is accesible via the last layer's ```out``` attribute.
    
  * ```update```: Updates every module in ```layers```.

* ```train_SUFFIX```: Trains a network with gradient descent.

  Required arguments:
   * ```seq``` (```Sequential_SUFFIX```): Sequential module to train.
   * ```inp``` (```float*```): Pointer to the input data.
   * ```targ``` (```float*```): Pointer to the target data.
   * ```bs``` (```int```): Batch size.
   * ```n_in``` (```int```): Number of input features.
   * ```n_epochs``` (```int```):  Number of epochs.
  
For end-to-end training with speed benchmarks, please run ```main.cpp``` or ```main.cu``` for the CPU and GPU respectively.
