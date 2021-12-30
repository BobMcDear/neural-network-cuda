# Neural network in CUDA/C++

## Description
This is an implementation of a neural net, completely from scratch (including basic tensor operations like matrix multiplication), in CUDA/C++. You can find the accompanying blog for the code [here](https://medium.datadriveninvestor.com/implementing-a-neural-net-in-cuda-from-scratch-part-1-introduction-9cd7f63573a7).

## Usage
The code is by no means efficient, so it is not a practical option and is meant for education purposes only. Nonetheless, here is an overview of the various classes & functions:

### Deep Learning Modules
Everything is implemented in both pure C++ (under CPU/) and CUDA/C++ (under GPU/). The syntax remains virtually identical, and there are only two points to bear in mind when switching between C++ and CUDA/C++:

1. C++ and CUDA/C++ modules end with the suffixes ```CPU``` and ```GPU``` respectively.
2. Don't forget to allocate and destroy CUDA arrays via ```cudaMallocManaged``` and ```cudaFree```

* ```linear.h/Linear_SUFFIX```: 
  * Initiation: 

    Required arguments: ```_bs``` (```int```, batch size), ```_n_in``` (```int```, number of input features), ```_n_out``` (```int```, number of output features)
    Optional argument: ```_lr``` (```float```, learning rate)
    
  * ```forward```: Runs a linear forward pass (weights set with Kaiming initialization, biases set to zero)

    Required arguments: ```_inp``` (```float*```, the input data), ```_out``` (```float*```, holds the output)
    
  * ```update```: Stores a copy of the weights for later use, then updates them as well as the biases

  * ```backward```: Stores the gradients of the loss with respect to the input in ```_inp```, assuming ```_out``` contains the gradients of the loss with respect to the next layer's input (i.e. the next layer has called ```backward```). The weights used are the ones saved during ```update```, and the copies are deleted thereafter.

* ```relu.h/ReLU_SUFFIX```:
  * Initiation:

    Required argument: ```_sz_out``` (```int```, the number of elements it's given)
    
  * ```forward```, ```backward```: Like ```Linear_SUFFIX``` but for ReLU

* ```mse.h/MSE_SUFFIX```:
  * Initiation: Like ReLU

  * ```forward```: Stores the predicted & target values for later use

    Required arguments: ```_inp``` (```float*```, the predicted values), ```_out``` (```float*```, the target values)
    
  * ```_forward```: Calculates the MSE but does not store the predicted & target values, meaning ```forward``` must be called regardless of ```_forward```

    Required arguments: Like ```MSE_SUFFIX``` but ```_out``` must have an extra element because the MSE will be saved in ```_out[sz_out]```
    
  * ```backward```: Stores the gradients of the target values with respect to the predicted values in ```_inp```

* ```sequential.h/Sequential_SUFFIX```:

  * Initiation: 

    Required arguments: ```layers``` (```std::vector<Module*>```, layers to be sequenced)
    
  * ```forward```: Feeds the input to the first layer, the output of that to the second layer, ...

    Required arguments: ```inp``` (```float*```, the input data), ```out```, (```float*```, there for consistency and doesn't get used. The output is accesible via the last layer's ```out``` attribute)
    
  * ```update```: Goes through ```layers``` in reverse and calls their ```update``` & ```backward``` methods (first the last layer's ```update```, then its ```backward```, then the second-to-last layer's ```update```, then its ```backward```, ...)

* ```train_SUFFIX```: Trains a network with gradient descent

  Required arguments: ```seq``` (```Sequential_SUFFIX```, the network), ```inp``` (```float*```, the input data), ```targ``` (```float*```, the target data), ```bs``` (```int```, batch size), ```n_in``` (```int```, number of input features), ```n_epochs```, (```int```, number of epochs)
  
For end-to-end training with speed benchmakrs, please run ```main.cpp``` or ```main.cu``` for the CPU and GPU respectively.
