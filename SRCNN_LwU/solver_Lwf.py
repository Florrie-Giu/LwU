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
        self.experiment_number = config.experiment  # 实验情况
        self.psnr_loss = config.psnr_loss  # 使用psnr值来计算梯度（True or False）
        # self.add_freq = config.add_freq                     # 增加频域loss（True or False)
        self.training_loader = training_loaders
        self.testing_loader = testing_loaders
        self.name = "SRCNN_LwU_init_BSD.pth"
        self.checkpoints = "./SRCNN_LwU/checkpoints/"
        self.dir = './SRCNN_LwU/experiment/srcnn_icarl_task45/'
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
            self.optimizer = local_adam(self.model.parameters(), self.reg_lambda,
                                        self.lr)  # local_sgd(self.model.parameters(), self.reg_lambda, self.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)
        self.si_init()

    def si_init(self):
        self.task_count = 0
        self.damping_factor = 0.1
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

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
            if (epoch == 1 or batch_num == 0):
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
            progress_bar(batch_num, len(training_loader),
                         'Loss: %.4f, ewc: %.4f, ' % (train_loss / (batch_num + 1), ewc_loss))

        print("  Average Loss: {:.4f}".format(train_loss / len(training_loader)))

    def si_update_model(self, inputs, targets, tasks):

        unreg_gradients = {}

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        out = self.model(inputs)
        loss = self.criterion(out, targets)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets)

        reg_loss = 0
        # Normal update with regularization
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists

            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    a = (p - task_param[n]) + 1e-4
                    task_reg_loss += (importance[n] * a ** 2).sum()
                reg_loss += task_reg_loss
                print("reg_loss: {:.9f}".format(reg_loss * self.reg_lambda))

            if (reg_loss > 10 and reg_loss <= 100):
                self.reg_lambda = 1e-1
            elif (reg_loss > 100 and reg_loss <= 1e3):
                self.reg_lambda = 1e-2
            elif (reg_loss > 1e3):
                self.reg_lambda = 1e-5

            if (reg_loss > 1e-2 and reg_loss <= 1):
                self.reg_lambda = 1e+1
            elif (reg_loss > 1e-4 and reg_loss <= 1e-3):
                self.reg_lambda = 1e+2
            elif (reg_loss < 1e-5):
                self.reg_lambda = 1e+5

            if (reg_loss < 0):
                self.reg_lambda = 0
            loss += self.reg_lambda * reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss, reg_loss, out

    def si_train(self, task_no, epoch, training_loader):

        self.model.train()

        train_loss = 0

        for batch_num, (data, target, _) in enumerate(training_loader):
            data, target = data.to(self.device), target.to(self.device)
            loss, reg_loss, sr = self.si_update_model(data, target, task_no)

            train_loss += loss.item()

            # self.optimizer.step(self.model.reg_params, self.device)
            # progress_bar(batch_num, len(training_loader), 'Loss: %.4f, reg_loss: %.4f' % (train_loss / (batch_num + 1), reg_loss))

        print(" Average Loss: {:.4f}".format(train_loss / len(training_loader)))

    def _calc_fisher_information(self, images, labels):
        self.optimizer.zero_grad()
        mle = self.criterion(self.model(images), labels)
        mle.backward()

        params = []
        for param in self.model.parameters():
            params.append(self.reg_lambda * param.grad ** 2)

        return params

    def icarl_train(self, epoch, training_loader):

        self.model.train()
        train_loss = 0

        for batch_num, (data, target, _) in enumerate(training_loader):
            self.criterion_dil = DilLoss(self.upscale_factor, data.size(0))

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            sr = self.model(data)

            mle = self.criterion(sr, target)
            dil_loss = self.criterion_dil(sr, target)
            print('dil_loss', dil_loss.item())

            if(dil_loss.item() <0):
                self.reg_lambda = 0
            # elif(dil_loss>1e-3 and dil_loss<1e-1):
            #     self.reg_lambda = 1e+1
            # elif (dil_loss > 1e-5 and dil_loss <= 1e-3):
            #     self.reg_lambda = 1e+2
            # elif (dil_loss <= 1e-5):
            # else:
            #     self.reg_lambda = 1e+3

            loss = mle + self.reg_lambda * dil_loss
            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            # self.optimizer.step(self.model.reg_params, self.device)
            progress_bar(batch_num, len(training_loader),
                         'Loss: %.4f, dil_loss: %.4f, ' % (train_loss / (batch_num + 1), dil_loss.item()*self.reg_lambda))

        print("  Average Loss: {:.4f}".format(train_loss / len(training_loader)))

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
                    prediction = prediction[:, :, :target.size(-2), :target.size(-1)]
                if prediction.size(-2) < target.size(-2) or prediction.size(-1) < target.size(-1):
                    print("the dimention of hr image is not equal to sr's! ")
                    target = target[:, :, :prediction.size(-2), :prediction.size(-1)]

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
                    f.write('task {} ] psnr : {:.4f}, ssim: {:.4f}'.format(task_no, psnr, ssim) + "\n")

        print(" task {} , Average PSNR: {:.4f} dB, SSIM: {:.4f}".format(task_no, avg_psnr / len(testing_loader),
                                                                        avg_ssim / len(testing_loader)))
        with open(os.path.join(self.dir, 'testpsnr.txt'), 'a') as f:
            f.write('[epoch {} / task {} ] avg_psnr : {:.4f}, avg_ssim: {:.4f}'.format(epoch, task_no,
                                                                                       avg_psnr / len(testing_loader),
                                                                                       avg_ssim / len(
                                                                                           testing_loader)) + "\n")

    def _make_dir(self, path):
        if not os.path.exists(path): os.makedirs(path)

    def run(self, dataname):

        # 建立模型
        self.build_model()
        self._make_dir(self.dir)
        self.dir = os.path.join(self.dir, 'init_' + dataname)
        print(self.dir)
        self._make_dir(self.dir)
        self._make_dir(self.dir + '/model')

        if self.config.model_path != '':
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=lambda storage, loc: storage))
            print('load model successful ! ')

        # 第一次训练
        # self.model.init_reg_params()

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train(self.training_loader)
            self.scheduler.step(epoch)
            if epoch % 500 == 0:
                self.test(epoch, task_no=0, testing_loader=self.testing_loader)
                self.save_model('srcnn_init_' + str(epoch) + '.pth')
            if epoch == self.nEpochs:
                # 创建一个Ω更新的优化器类，传入到模型当中更新参数的Ω值
                optimizer_ft = omega_update(self.model.reg_params)
                # 更新Ω
                self.model = compute_omega_grads_norm(self.model, self.training_loader, optimizer_ft, self.device)
                self.save_model('srcnn_init_' + dataname + '.pth')

    def mutli_run(self, num_of_tasks):

        # 建立模型
        self.build_model()
        self._make_dir(self.dir)

        if self.config.model_path != '':
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=lambda storage, loc: storage))
            print("load model from {} is successful !".format(self.config.model_path))

        dir = self.dir

        for task_no in range(4, num_of_tasks + 1):
            print("Training the model on task {}".format(task_no))
            # 把每一个任务对应的数据集取出来
            dataloader_train = self.training_loader[task_no - 1]
            dataloader_test = self.testing_loader[task_no - 1]

            self.dir = os.path.join(dir, 'task_' + str(task_no))
            print(self.dir)
            self._make_dir(self.dir)
            self._make_dir(self.dir + '/model')

            # 在训练之前先测试一次
            # self.test(0, task_no, dataloader_test)

            # 实验2：增益训练，每个task都有一个baseline值和(nEpochs-1)个psnr值，除第一个任务外
            for epoch in range(4, self.nEpochs + 1):

                print("\n===> Epoch {} starts:".format(epoch))
                self.icarl_train(epoch, dataloader_train)
                self.scheduler.step(epoch)
                if epoch % 50 == 0 or epoch == self.nEpochs:
                    self.test(epoch, task_no, dataloader_test)
                    self.save_epoch('srcnn_task_' + str(task_no) + '_' + str(epoch) + '.pth')

                # if epoch == self.nEpochs:
                #     self.test(epoch, task_no, dataloader_test)
                #     # 创建一个Ω更新的优化器类，传入到模型当中更新参数的Ω值
                #     optimizer_ft = omega_update(self.model.reg_params)
                #     # 更新Ω
                #     self.model = compute_omega_grads_norm(self.model, dataloader_train, optimizer_ft, self.device)
                #     self.save_model('srcnn_task_'+str(task_no)+'.pth')

            # if task_no>1:
            #     self.model = consolidate_reg_params(self.model)
            # self._si_more_opt(dataloader_train)

    def _si_more_opt(self, train_loader):
        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self._calc_importance(train_loader)
        # Save the weight and importance of weights of current task
        self.task_count += 1
        if len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance': importance, 'task_param': task_param}

    def _calc_importance(self, dataloader):
        # Initialize the importance matrix
        if len(self.regularization_terms) > 0:  # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
            self.w[n].zero_()

        return importance

    def print_network(self):
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
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
            index = len(lines) - 1 - lines[::-1].index("task" + str(task_no) + "\n")
            if index != -1:
                old_psnr = lines[index + self.nEpochs].split('\n')[0]
        elif experiment_no == 2:
            if task_no == num_of_tasks:
                old_psnr = lines[-1].split('\n')[0]
            else:
                index = len(lines) - 1 - lines[::-1].index("task" + str(task_no + 1) + "_baseline" + "\n")
                if index != -1:
                    old_psnr = lines[index - 1].split('\n')[0]

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



