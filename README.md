# LwU
Learning while Using : Defying forgetting in Medical Super-Resolution Tasks

Description of the files in this repository
---------------------------------------------------
For DRN_LwU
1) ``main_drn.py``: Execute this file to train the drn model on the task sequence
2) ``/DRN_LwU/option.py``: Contains the parameter selection for training the drn model, which can specify which continuous learning algorithm, such as EWC, SI, iCaRL-CNN, LwU
3) ``/DRN_LwU/train.py``: Contains the function to train the model
4) ``/DRN_LwU/data``: Contains the data generator
5) ``/DRN_LwU/model``: contains the definition of the model
6) ``utils/lwu_utils.py``: Utilities for the optimizers that implement the idea of computing the gradients       							locally
7) ``optimizer_adam.py``: This file contains the optimizer classes, that realize the idea of computing the 								 gradients of the penalty term of the loss function locally 
8) ``optimizer_lib.py``: This file contains the optimizer classes, that realize the idea of computing the 								 gradients of the penalty term of the loss function locally 


For SRCNN_LwU
1) ``/main_srcnn.py``: Execute this file to train the srcnn model on the task sequence
2) ``/SRCNN_LwU/option.py``: Contains the parameter selection for training the drn model, which can specify which continuous learning algorithm, such as EWC, SI, iCaRL-CNN, LwU
3) ``/SRCNN_LwU/train.py``: Contains the function to train the model
4) ``/SRCNN_LwU/data``: Contains the data generator
5) ``/model``: contains the definition of the model
6) ``utils/lwu_utils.py``: Utilities for the optimizers that implement the idea of computing the gradients       							locally
7) ``optimizer_adam.py``: This file contains the optimizer classes, that realize the idea of computing the 								 gradients of the penalty term of the loss function locally 
8) ``optimizer_lib.py``: This file contains the optimizer classes, that realize the idea of computing the 								 gradients of the penalty term of the loss function locally 

Training
------------------------------

To begin the training process on the sequence of tasks, use the **`main_drn.py`** file. Simply execute the following lines to begin the training process

```sh
python main_drn.py --lml lwu --reg_lambda 0.1
```

The file takes the following arguments
* **data_dir**: Tasks training dataset directory
* **data_test**: Test dataset name
* **pre_train**: Pre-trained model directory **Default**:'./premodel/model_div2k.pt'
* ***use_gpu***: Set the flag to true to train the model on the GPU **Default**: False
* ***batch_size***: Batch Size. **Default**: 8
* ***epochs***: Number of epochs you want to train the model for. **Default**: 200
* **test_every**: Do test per every N batches. **Default**:50
* ***lr***: Initial learning rate for the model. The learning rate is decayed every 20th epoch.**Default**: 1e-4
* ***reg_lambda***: The regularization parameter that provides the trade-off between the cross entropy loss function and the penalty for changes to important weights. **Default**: 0.01
* **lml**: Lifelong machine learning method, options=[ewc, si, icarl-cnn, lwu]
* **test_only**:Set this option to test the model
* **save**: File name to save in the 'expriment' folder



# Acknowledgements
This code is built on [DRN](https://github.com/guoyongcs/DRN), [MAS-PyTorch](https://github.com/wannabeOG/MAS-PyTorch), [iCaRL](https://github.com/srebuffi/iCaRL) and [Continual Learning Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark). We thank the authors for sharing their codes.
