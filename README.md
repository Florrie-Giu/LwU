# LwU
Learning while Using : Defying forgetting in Medical Super-Resolution Tasks

Description of the files in this repository
---------------------------------------------------
1) ``main.py``: Execute this file to train the model on the sequence of tasks
2) ``mas.py``: Contains functions that help in training and evaluating the model on these tasks (the forgetting <				that is undergone by the model)
3) ``model_class.py``: Contains the classes defining the model
4) ``model_train.py``: Contains the function that trains the model
5) ``optimizer_lib.py``: This file contains the optimizer classes, that realize the idea of computing the 								 gradients of the penalty term of the loss function locally 
6) ``data_prep.py``: File to download the datset and split the dataset into 4 folders that are interpreted as 						 different tasks 
7) ``utils/model_utils.py``: Utilities for training the model on the sequence of tasks
8) ``utils/mas_utils.py``: Utilities for the optimizers that implement the idea of computing the gradients       							locally

Training
------------------------------

To begin the training process on the sequence of tasks, use the **`main.py`** file. Simply execute the following lines to begin the training process

```sh
python3 main.py
```

The file takes the following arguments

* ***use_gpu***: Set the flag to true to train the model on the GPU **Default**: False
* ***batch_size***: Batch Size. **Default**: 8
* ***num_freeze_layers***: The number of layers in the feature extractor (features) of an Alexnet model, that you want to train. The rest are frozen and they are not trained. **Default**: 2
* ***num_epochs***: Number of epochs you want to train the model for. **Default**: 10
* ***init_lr***: Initial learning rate for the model. The learning rate is decayed every 20th epoch.**Default**: 0.001 
* ***reg_lambda***: The regularization parameter that provides the trade-off between the cross entropy loss function and the penalty for changes to important weights. **Default**: 0.01



# Acknowledgements
This code is built on [MAS-PyTorch](https://github.com/wannabeOG/MAS-PyTorch). We thank the authors for sharing their codes of MAS PyTorch version(https://github.com/wannabeOG/MAS-PyTorch).
