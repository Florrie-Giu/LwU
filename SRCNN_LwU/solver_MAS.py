from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn
import os
# import imageio
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from SRCNN_LwU.model import Net
from SRCNN_LwU.progress_bar import progress_bar
from SRCNN_LwU.optimizer_lib import *
from SRCNN_LwU.optimizer_adam import *
from SRCNN_LwU.mas_utils import *
from SRCNN_LwU.lossFun import *

class SRCNNTrainer(object):
    def __init__(self, config, training_loaders, testing_loaders, consolidate=True):
        super(SRCNNTrainer, self).__init__()
        self.consolidate = consolidate
        self.fisher_estimation_sample_size = config.batchSize
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.reg_lambda = config.reg_lambda
        self.experiment_number = config.experiment   # 实验情况
        self.psnr_loss = config.psnr_loss                   # 使用psnr值来计算梯度（True or False）
        # self.add_freq = config.add_freq                     # 增加频域loss（True or False)
        self.training_loader = training_loaders
        self.testing_loader = testing_loaders
        self.name = "SRCNN_LwU_init_BSD.pth"
        self.checkpoints = "./SRCNN_LwU/checkpoints/"
        self.dir = './SRCNN_LwU/experiment/srcnn_ewc_lambda1e7/'
        self.config = config

    def build_model(self):
        # 获得模型
        self.model = Net(num_channels=3, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)

        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        # 自定义带正则项的优化器
        if self.config.lml == 'lwu':
            self.optimizer = local_adam(self.model.parameters(), self.reg_lambda, self.lr) #local_sgd(self.model.parameters(), self.reg_lambda, self.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self, model_name):
        model_out_path = os.path.join(self.dir, 'model', model_name)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def save_epoch(self, model_name):
        model_out_path = os.path.join(self.dir, 'model', model_name)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved the epoch{}".format(model_out_path))

    def train(self, epoch, training_loader):

        self.model.train()
        train_loss = 0
        images, labels = None, None
        init_parameters = []
        for batch_num, (data, target, _) in enumerate(training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            mle = self.criterion(self.model(data), target)

            # ewc 新加的损失
            if(epoch==1 or batch_num==0):
                ewc_loss = 0
                images = data
                labels = target
                init_parameters = self.model.init_params()
            else:
                fisher_information = self._calc_fisher_information(images, labels)
                ewc_penalty_list = []
                for f, p1, p2 in zip(fisher_information, init_parameters, self.model.get_params()):
                    a = (p1 - p2) ** 2 + 1e-4
                    ewc_penalty_list.append(torch.sum(torch.mul(f, a)))
                ewc_loss = torch.stack(ewc_penalty_list).sum()

            loss = mle + ewc_loss
            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            # self.optimizer.step(self.model.reg_params, self.device)
            progress_bar(batch_num, len(training_loader), 'Loss: %.4f, ewc: %.4f, ' % (train_loss / (batch_num + 1), ewc_loss))

        print("  Average Loss: {:.4f}".format(train_loss / len(training_loader)))


    def _calc_fisher_information(self, images, labels):
        self.optimizer.zero_grad()
        mle = self.criterion(self.model(images), labels)
        mle.backward()

        params = []
        for param in self.model.parameters():
            params.append(self.reg_lambda * param.grad ** 2)

        return params


    def test(self, epoch, task_no, testing_loader):
        from SRCNN_LwU import pyssim
        if testing_loader == None:
            testing_loader = self.testing_loader
        self.model.eval()
        avg_psnr = 0
        avg_ssim = 0

        mse = torch.nn.MSELoss()
        with torch.no_grad():
            for batch_num, (data, target, filename) in enumerate(testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                # print('size', data.size(), 'and', target.size())
                prediction = self.model(data)

                if prediction.size(-2) > target.size(-2) or prediction.size(-1) > target.size(-1):
                    print("the dimention of sr image is not equal to hr's! ")
                    prediction = prediction[:,:,:target.size(-2),:target.size(-1)]
                if prediction.size(-2) < target.size(-2) or prediction.size(-1) < target.size(-1):
                    print("the dimention of hr image is not equal to sr's! ")
                    target = target[:,:,:prediction.size(-2),:prediction.size(-1)]

                mseVal = mse(prediction, target)
                psnr = 10 * log10(1 / mseVal.item())
                avg_psnr += psnr
                ssim = pyssim.ssim(prediction, target)
                avg_ssim += ssim


                self._make_dir(os.path.join(self.dir, 'results'))
                out = prediction.cpu()
                out = out.squeeze(0)
                img = ToPILImage()(out)
                img.save(os.path.join(self.dir, 'results', filename[0]))

                # progress_bar(batch_num, len(testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
                with open(os.path.join(self.dir, 'testpsnr.txt'), 'a') as f:
                    f.write('task {} ] psnr : {:.4f}, ssim: {:.4f}'.format(task_no, psnr, ssim)+"\n")

        print(" task {} , Average PSNR: {:.4f} dB, SSIM: {:.4f}".format(task_no, avg_psnr/len(testing_loader), avg_ssim/len(testing_loader)))
        with open(os.path.join(self.dir, 'testpsnr.txt'), 'a') as f:
            f.write('[epoch {} / task {} ] avg_psnr : {:.4f}, avg_ssim: {:.4f}'.format(epoch, task_no,avg_psnr/len(testing_loader),avg_ssim/len(testing_loader))+"\n")

    def _make_dir(self, path):
        if not os.path.exists(path): os.makedirs(path)

    def run(self, dataname):

        # 建立模型
        self.build_model()
        self._make_dir(self.dir)
        self.dir = os.path.join(self.dir, 'init_'+dataname)
        print(self.dir)
        self._make_dir(self.dir)
        self._make_dir(self.dir + '/model')

        if self.config.model_path !='':
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=lambda storage, loc: storage))
            print('load model successful ! ')

        # 第一次训练
        self.model.init_reg_params()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train(self.training_loader)
            self.scheduler.step(epoch)
            if epoch % 500 == 0:
                self.test(epoch, task_no=0, testing_loader=self.testing_loader)
                self.save_model('srcnn_init_'+str(epoch)+'.pth')
            if epoch == self.nEpochs:
                # 创建一个Ω更新的优化器类，传入到模型当中更新参数的Ω值
                optimizer_ft = omega_update(self.model.reg_params)
                # 更新Ω
                self.model = compute_omega_grads_norm(self.model, self.training_loader, optimizer_ft, self.device)
                self.save_model('srcnn_init_'+dataname+'.pth')

    def mutli_run(self, num_of_tasks):

        # 建立模型
        self.build_model()
        self._make_dir(self.dir)

        if self.config.model_path != '':
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=lambda storage, loc: storage))
            print("load model from {} is successful !" .format(self.config.model_path))

        dir = self.dir

        for task_no in range(1, num_of_tasks + 1):
            print("Training the model on task {}".format(task_no))
            # 把每一个任务对应的数据集取出来
            dataloader_train = self.training_loader[task_no - 1]
            dataloader_test = self.testing_loader[task_no - 1]

            self.dir = os.path.join(dir, 'task_' + str(task_no))
            print(self.dir)
            self._make_dir(self.dir)
            self._make_dir(self.dir + '/model')

            # 在训练之前先测试一次
            self.test(0, task_no, dataloader_test)

            # 实验2：增益训练，每个task都有一个baseline值和(nEpochs-1)个psnr值，除第一个任务外
            for epoch in range(1, self.nEpochs+1):

                print("\n===> Epoch {} starts:".format(epoch))
                self.train(epoch, dataloader_train)
                self.scheduler.step(epoch)
                if epoch % 50 == 0 or epoch == self.nEpochs:
                    self.test(epoch, task_no, dataloader_test)
                    self.save_epoch('srcnn_task_'+str(task_no)+'_'+str(epoch)+'.pth')

                # if epoch == self.nEpochs:
                #     self.test(epoch, task_no, dataloader_test)
                #     # 创建一个Ω更新的优化器类，传入到模型当中更新参数的Ω值
                #     optimizer_ft = omega_update(self.model.reg_params)
                #     # 更新Ω
                #     self.model = compute_omega_grads_norm(self.model, dataloader_train, optimizer_ft, self.device)
                #     self.save_model('srcnn_task_'+str(task_no)+'.pth')

            # if task_no>1:
            #     self.model = consolidate_reg_params(self.model)


    def print_network(self):
        num_params = 0
        for param in self.model.parameters():
            num_params+=param.numel()
        print(self.model)
        print('Total number of parameters: %d' % num_params)

    def onlytest(self, model_path):
        print('===> Building model')
        self.build_model()
        self.print_network()
        self._make_dir(self.dir)
        if model_path != '':
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            print('Pre-trained SRCNN model is loaded.')
        self.test(epoch=0, task_no=1, testing_loader=self.testing_loader)

    def compute_forgetting(self, task_no, experiment_no, num_of_tasks, dataloader, performance_path, model_path):
        """
        函数compute_forgetting从存储在特定于任务的文件夹中的文件（performance.txt)中
        读取先前的性能，并将其与该任务上模型的当前性能进行比较。
        Inputs
        1) task_no: The task number on which you want to compute the forgetting
        2) dataloader: The dataloader that feeds in the data to the model

        Outputs
        1) forgetting: The amount of forgetting undergone by the model

        Function: Computes the "forgetting" that the model has on the
        """
        file_object = open(performance_path)
        lines = file_object.readlines()  # 读取全部内容
        old_psnr = 0
        # 不同实验记录的txt值不一样
        if experiment_no == 1:
            # 取最后一次出现的
            index = len(lines) - 1 - lines[::-1].index("task"+str(task_no)+"\n")
            if index != -1:
                old_psnr=lines[index+self.nEpochs].split('\n')[0]
        elif experiment_no == 2:
            if task_no == num_of_tasks:
                old_psnr = lines[-1].split('\n')[0]
            else:
                index = len(lines) - 1 - lines[::-1].index("task" + str(task_no+1)+"_baseline" + "\n")
                if index != -1:
                    old_psnr = lines[index-1].split('\n')[0]

        print("old psnr value ", old_psnr)

            # load the model for inference
        # model = model_inference(task_no, use_gpu=False)
        self.build_model()
        self.print_network()
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        print('Pre-trained SRCNN model is loaded.')

        self.model.eval()
        avg_psnr = 0
        mse = torch.nn.MSELoss()
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mseVal = mse(prediction, target)
                psnr = 10 * log10(1 / mseVal.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(dataloader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        avg_psnr /= len(dataloader)
        old_psnr = float(old_psnr)
        forgetting = avg_psnr - old_psnr
        return forgetting



