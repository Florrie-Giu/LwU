import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable, functional
from .mas_utils import get_data_loader
from torch.nn import functional as F



class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor=4):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )

        self.reg_params = None

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def init_reg_params(self):
        reg_params = {}

        for name, param in self.named_parameters():

            print("Initializing omega values for layer", name)
            omega = torch.zeros(param.size())

            init_val = param.data.clone()
            param_dict = {}

            # for first task, omega is initialized to zero
            param_dict['omega'] = omega
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

        self.reg_params = reg_params

    def init_reg_params_across_tasks(self):
        reg_params = self.reg_params

        for name, param in self.named_parameters():
            print("Initializing omega values for layer for the new task", name)
            # 获取该参数的字典
            param_dict = reg_params[param]

            # Store the previous values of omega
            prev_omega = param_dict['omega']

            # Initialize a new omega
            new_omega = torch.zeros(param.size())

            init_val = param.data.clone()

            # 多加了一个前Ω的属性
            param_dict['prev_omega'] = prev_omega
            # 再把当前任务的Ω初始化为0
            param_dict['omega'] = new_omega

            # 存储参数在前一个任务训练后保留下来的值，作为该任务的初始值
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

        self.reg_params = reg_params


    def init_params(self):
        params = []
        for param in self.parameters():
            params.append(param.detach())
        return params

    def get_params(self):
        params = []
        for param in self.parameters():
            params.append(param)
        return params

    @property
    def name(self):
        return (
            'Net'
            '-reg_params{reg_params}'
        ).format(
            reg_params=self.reg_params,
        )


    def estimate_fisher(self, x, y, optimizer, batch_size=1):
        # sample loglikelihoods from the dataset. (采样对数似然)
        # data_loader = get_data_loader(dataset, batch_size)
        '''
        loglikelihoods = []
        for x, y,_ in data_loader:

            x = x.cuda() if self._is_on_cuda() else Variable(x)
            y = y.cuda() if self._is_on_cuda() else Variable(y)

            # ls = criterion(self(x), y)
            # # log f(x)
            # loglikelihoods.append(
            #     autograd.grad(ls, self.parameters(), create_graph=True, retain_graph=True)[0]
            # )
            loglikelihoods.append(criterion(self(x), y))

            if len(loglikelihoods) >= sample_size // batch_size:
                break

        # loglikelihood_grads = [torch.stack(gs) for gs in loglikelihoods]

        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()

        # s(thate) = (log f(x))' 对thate求导
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])

        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        #fisher对角矩阵 = 梯度的平方再求期望
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        '''

        criterion = torch.nn.MSELoss()
        optimizer.zero_grad()
        preds = self(x)
        loss = criterion(preds, y)
        loss.backward()

        fisher_diagonals = []
        for param in self.parameters():
            fisher_diagonals.append(self.reg_params * param.grad ** 2)


        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        print(param_names)
        print(len(fisher_diagonals))

        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            # if not 'bias' in n:
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'.format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.reg_params/2)*sum(losses)

        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)    # 正态分布
        m.bias.data.zero_()   # 为零
