'''*-coding:utf-8 *-
 @Time     : 2020/6/2314:11
 @Author   : florrie(zfh)
'''

#!/usr/bin/env python
# coding: utf-8

import torch
torch.backends.cudnn.benchmark=True

from torch.utils.data import DataLoader

import argparse

#lwu
# from SRCNN_LwU.solver_MAS import *
# ewc or lwf
from SRCNN_LwU.solver_Lwf import *
from SRCNN_LwU.data.data_loader import get_training_set_MAS, get_test_set_MAS, get_testdata_MAS
from SRCNN_LwU.mas_utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--train_dir', type=str, default="./origin_task_MAS", help='training path')
parser.add_argument('--test_dir', type=str, default="./origin_task_MAS", help='testing path')
parser.add_argument('--model_path', type=str, default='./SRCNN_LwU/experiment/srcnn_icarl/task_3/model/srcnn_task_3_200.pth',
                                            help='model path. pretrain:/premodel/srcnn_div2k_200.pth,'
                                                 'test: ./SRCNN_LwU/experiment/srcnn_ewc/task_5/model/srcnn_task_5_200.pth')
parser.add_argument('--batchSize', type=int, default=5, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train', type=bool, default=True, help='is the model traned? Default=True')


parser.add_argument('--lml', type=str, default='ewc', help='Learning Rate. Default=0.01')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srcnn', help='choose which model is going to use')
parser.add_argument('--reg_lambda', type=float, default=1e-2, help='Regularization parameter： ewc=1e+8, lwf=1, lwu=0.01, icarl=1e+3')
parser.add_argument('--psnr_loss', type=bool, default=True, help='use the psnr to compute the grad')
parser.add_argument('--experiment', type=int, default=1, help='1 represent the normal training, '
                                                                     '2 represent the gain training')
parser.add_argument('--add_freq', type=bool, default=False, help='increase the loss which constrains SR and HR to be consistent in frequency domain')

args = parser.parse_args()


def main():
    '''多任务数据集的制作'''
    data_dir = "./origin_task_MAS"

    dloaders_train = []
    dloaders_test = []
    # 制作多任务数据集
    print('===> Loading datasets')
    for tdir in sorted(os.listdir(data_dir)):  # 获取该目录下的子文件
        # 获得目录名
        print("====> dataset name of task is ", tdir)
        train_dir = os.path.join(data_dir, tdir)
        test_dir = os.path.join(data_dir, tdir)

        train_set = get_training_set_MAS(args.upscale_factor, train_dir)
        test_set = get_testdata_MAS(args.upscale_factor, test_dir)
        training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
        testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

        # append the dataloaders of these tasks
        dloaders_train.append(training_data_loader)  # list{}, dloaders_train[0]=dataloader
        dloaders_test.append(testing_data_loader)

    print('===> Load finish!')
    # get the number of tasks in the sequence
    num_of_tasks = len(dloaders_train)

    model = SRCNNTrainer(args, dloaders_train, dloaders_test)
    
    '''初始化模型的单数据集训练
    train_set = get_training_set_MAS(args.upscale_factor, args.train_dir, hr_size=192)
    test_set = get_testdata_MAS(args.upscale_factor, args.test_dir)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    model = SRCNNTrainer(args, training_data_loader, testing_data_loader)'''
    
    if args.train:
        # model.run("bsd")
        # initialize the parameters.
        # xavier_initialize(model)
        model.mutli_run(num_of_tasks)
    else:
        model.onlytest(args.model_path)

if __name__ == '__main__':
    main()









