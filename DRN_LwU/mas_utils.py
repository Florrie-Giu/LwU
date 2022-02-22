#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil

import sys


def init_reg_params(model, use_gpu, freeze_layers = []):
	"""
	Input:
	1) model: A reference to the model that is being trained
	2) use_gpu: Set the flag to True if the model is to be trained on the GPU
	3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
		case of computational limitations where computing the importance parameters for the entire model
		is not feasible

	Output:
	1) model: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters is calculated and the updated
	model with these reg_params is returned.
	返回的模型：计算包含所有可训练参数的重要性加权（omega），
				init_val（保留对参数初始值的引用）的字典，
				并返回带有这些reg_params的更新模型。

	Function: Initializes the reg_params for a model for the initial task (task = 1)	
	
	"""
	device = torch.device("cuda:0" if use_gpu else "cpu")

	reg_params = {}

	for name, param in model.tmodel.named_parameters():
		if not name in freeze_layers: # 不在freeze_layer层里面的参数具有正则项
			
			print ("Initializing omega values for layer", name)
			omega = torch.zeros(param.size())
			omega = omega.to(device)

			init_val = param.data.clone()
			param_dict = {}

			#for first task, omega is initialized to zero
			param_dict['omega'] = omega
			param_dict['init_val'] = init_val

			#the key for this dictionary is the name of the layer
			reg_params[param] = param_dict

	model.reg_params = reg_params

	return model


def init_reg_params_across_tasks(model, use_gpu, freeze_layers = []):
	"""
	Input:
	1) model: A reference to the model that is being trained
	2) use_gpu: Set the flag to True if the model is to be trained on the GPU
	3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
		case of computational limitations where computing the importance parameters for the entire model
		is not feasible

	Output:
	1) model: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters is calculated and the updated
	model with these reg_params is returned.


	Function: Initializes the reg_params for a model for other tasks in the sequence (task != 1)	
	"""

	#Get the reg_params for the model 
	
	device = torch.device("cuda:0" if use_gpu else "cpu")

	reg_params = model.reg_params

	for name, param in model.tmodel.named_parameters():
		
		if not name in freeze_layers:

			if param in reg_params:
				# 获取该参数的字典
				param_dict = reg_params[param]
				# print ("Initializing the omega values for layer for the new task", name)
				
				#Store the previous values of omega
				prev_omega = param_dict['omega']
				
				#Initialize a new omega
				new_omega = torch.zeros(param.size())
				new_omega = new_omega.to(device)

				init_val = param.data.clone()
				init_val = init_val.to(device)
				# 多加了一个前Ω的属性
				param_dict['prev_omega'] = prev_omega
				# 再把当前任务的Ω初始化为0
				param_dict['omega'] = new_omega

				#store the initial values of the parameters
				# 存储参数在前一个任务训练后保留下来的值，作为该任务的初始值
				param_dict['init_val'] = init_val

				#the key for this dictionary is the name of the layer
				reg_params[param] = param_dict

	model.reg_params = reg_params

	return model


def consolidate_reg_params(model):
	"""
	Input:
	1) model: A reference to the model that is being trained
	2) use_gpu: Set the flag to True if you wish to train the model on a GPU

	Output:
	1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters


	Function: This function updates the value (adds the value) of omega across the tasks that the model is 
	exposed to
	
	"""
	#Get the reg_params for the model 
	reg_params = model.reg_params

	for name, param in model.named_parameters():
		if param in reg_params:
			param_dict = reg_params[param]
			# print ("Consolidating the omega values for layer", name)
			
			#Store the previous values of omega
			prev_omega = param_dict['prev_omega']
			new_omega = param_dict['omega']

			new_omega = torch.add(prev_omega, new_omega)
			del param_dict['prev_omega']
			
			param_dict['omega'] = new_omega

			#the key for this dictionary is the name of the layer
			reg_params[param] = param_dict

	model.reg_params = reg_params

	return model


def compute_omega_grads_norm(main_model, dual_models, dataloader, main_optimizer, dual_optimizers, device):
	"""
	计算Ωij的梯度正则
	Inputs:
	1) model: A reference to the model for which omega is to be calculated
	2) dataloader: A dataloader to feed the data to the model
	3) optimizer: An instance of the "omega_update" class
	4) use_gpu: Flag is set to True if the model is to be trained on the GPU

	Outputs:
	1) model: An updated reference to the model is returned

	Function: Global version for computing the l2 norm of the function (neural network's) outputs. In 
	addition to this, the function also accumulates the values of omega across the items of a task
	
	"""

	#Alexnet object
	main_model.eval()
	

	index = 0
	for batch, (inputs, labels, _) in enumerate(dataloader):

		# get the inputs and labels
		inputs = [a.to(device) for a in inputs]
		labels = labels.to(device)
		# print(labels.size())
		# Zero the parameter gradients
		main_optimizer.zero_grad()
		for i in range(len(dual_optimizers)):
			dual_optimizers[i].zero_grad()

		#get the function outputs
		# forward
		outputs = main_model(inputs[0])   # list{3}
		sr2lr = []
		for i in range(len(dual_models)):
			sr2lr_i = dual_models[i](outputs[i - len(dual_models)])
			sr2lr.append(sr2lr_i)

		l2_norm = 0
		sum_norm = 0
		for i in range(len(outputs)):
			l2_norm = torch.norm(outputs[i], 2, dim=1)**2
			sum_norm += torch.sum(l2_norm)

		l2_dual = 0
		sum_dual = 0
		# compute dual loss
		for i in range(len([2,4])):
			l2_dual = torch.norm(sr2lr[i], 2, dim=1)**2
			sum_dual += torch.sum(l2_dual)

		# compute total loss
		del l2_norm
		del l2_dual
		
		sum_all = sum_norm + 0.1 * sum_dual

		#compute gradients for these parameters
		sum_all.backward()

		#optimizer.step computes the omega values for the new batches of data
		main_optimizer.step(main_model.reg_params, index, labels.size(0), device)
		for i in range(len(dual_optimizers)):
			dual_optimizers[i].step(dual_models[i].reg_params, index, labels.size(0), device)
		del labels
		
		index = index + 1

	return main_model, dual_models


#need a different function for grads vector
def compute_omega_grads_vector(model, dataloader, optimizer, use_gpu):
	"""
	计算梯度向量
	Inputs:
	1) model: A reference to the model for which omega is to be calculated
	2) dataloader: A dataloader to feed the data to the model
	3) optimizer: An instance of the "omega_update" class
	4) use_gpu: Flag is set to True if the model is to be trained on the GPU

	Outputs:
	1) model: An updated reference to the model is returned

	Function: This function backpropagates across the dimensions of the  function (neural network's) 
	outputs. In addition to this, the function also accumulates the values of omega across the items 
	of a task. Refer to section 4.1 of the paper for more details regarding this idea
	
	"""

	#Alexnet object
	model.tmodel.train(False)
	model.tmodel.eval(True)

	index = 0

	for dataloader in dset_loaders:
		for data in dataloader:
			
			#get the inputs and labels
			inputs, labels = data

			if(use_gpu):
				device = torch.device("cuda:0")
				inputs, labels = inputs.to(device), labels.to(device)

			#Zero the parameter gradients
			optimizer.zero_grad()

			#get the function outputs
			outputs = model.tmodel(inputs)

			for unit_no in range(0, outputs.size(1)):
				ith_node = outputs[:, unit_no]
				targets = torch.sum(ith_node)

				#final node in the layer
				if(node_no == outputs.size(1)-1):
					targets.backward()
				else:
					#This retains the computational graph for further computations 
					targets.backward(retain_graph = True)

				optimizer.step(model.reg_params, False, index, labels.size(0), use_gpu)
				
				#necessary to compute the correct gradients for each batch of data
				optimizer.zero_grad()

			
			optimizer.step(model.reg_params, True, index, labels.size(0), use_gpu)
			index = index + 1

	return model


#sanity check for the model to check if the omega values are getting updated
def sanity_model(model):
	'''
	检查模型完整性，检查Ω是否更新
	:param model:
	:return:
	'''
	for name, param in model.tmodel.named_parameters():
		
		print (name)
		
		if param in model.reg_params:
			param_dict = model.reg_params[param]
			omega = param_dict['omega']

			print ("Max omega is", omega.max())
			print ("Min omega is", omega.min())
			print ("Mean value of omega is", omega.min())



#function to freeze selected layers
def create_freeze_layers(model, no_of_layers = 2):
	"""
	Inputs
	1) model: A reference to the model
	2) no_of_layers: The number of convolutional layers that you want to freeze in the convolutional base of 
		Alexnet model. Default value is 2 

	Outputs
	1) model: An updated reference to the model with the requires_grad attribute of the 
			  parameters of the freeze_layers set to False 
	2) freeze_layers: Creates a list of layers that will not be involved in the training process

	Function: This function creates the freeze_layers list which is then passed to the `compute_omega_grads_norm`
	function which then checks the list to see if the omegas need to be calculated for the parameters of these layers  
	
	"""
	
	#The require_grad attribute for the parameters of the classifier layer is set to True by default 
	for param in model.tmodel.classifier.parameters(): # 全连接层的参数需要更新
		param.requires_grad = True

	for param in model.tmodel.features.parameters(): #	特征层的参数不进行更新
		param.requires_grad = False

	#return an empty list if you want to train the entire model
	if (no_of_layers == 0):
		return []

	temp_list = []
	freeze_layers = []

	#get the keys for the conv layers in the model
	for key in model.tmodel.features._modules:
		if (type(model.tmodel.features._modules[key]) == torch.nn.modules.conv.Conv2d):	# 如果是卷积层
			temp_list.append(key) # 把卷积层的序号放入list中（因为只有卷积层才有权重参数w和b）
	
	num_of_frozen_layers = len(temp_list) - no_of_layers # 5-2=3

	#set the requires_grad attribute to True for the layers you want to be trainable
	for num in range(0, num_of_frozen_layers):
		#pick the layers from the end
		temp_key = temp_list[num]
		
		for param in model.tmodel.features[int(temp_key)].parameters():
			param.requires_grad = True

		name_1 = 'features.' + temp_key + '.weight'
		name_2 = 'features.' + temp_key + '.bias'

		freeze_layers.append(name_1)
		freeze_layers.append(name_2)


	return [model, freeze_layers]




def model_inference(task_no, use_gpu=False):
	"""
	Inputs
	1) task_no: The task number for which the model is being evaluated
	2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value: False

	Outputs
	1) model: A reference to the model

	Function: Combines the classification head for a particular task with the shared model and
	returns a reference to the model is used for testing the process

	"""

	# all models are derived from the Alexnet architecture
	pre_model = models.alexnet(pretrained=True)
	model = shared_model(pre_model)

	path_to_model = os.path.join(os.getcwd(), "models")

	path_to_head = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))

	# get the number of classes by reading from the text file created during initialization for this task
	file_name = os.path.join(path_to_head, "classes.txt")
	file_object = open(file_name, 'r')
	num_classes = file_object.read()
	file_object.close()

	num_classes = int(num_classes)
	# print (num_classes)
	in_features = model.tmodel.classifier[-1].in_features

	del model.tmodel.classifier[-1]
	# load the classifier head for the given task identified by the task number
	classifier = classification_head(in_features, num_classes)
	classifier.load_state_dict(torch.load(os.path.join(path_to_head, "head.pth")))

	# load the trained shared model
	model.load_state_dict(torch.load(os.path.join(path_to_model, "shared_model.pth")))

	model.tmodel.classifier.add_module('6', nn.Linear(in_features, num_classes))

	# change the weights layers to the classifier head weights
	model.tmodel.classifier[-1].weight.data = classifier.fc.weight.data
	model.tmodel.classifier[-1].bias.data = classifier.fc.bias.data

	# device = torch.device("cuda:0" if use_gpu else "cpu")
	model.eval()
	# model.to(device)

	return model