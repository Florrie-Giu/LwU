import torch
from DRN_LwU import utility
from DRN_LwU.mas_utils import *
from DRN_LwU.optimizer_lib import *
from DRN_LwU.checkpoint import Checkpoint
from decimal import Decimal
from tqdm import tqdm
import os
from DRN_LwU.loss import DilLoss


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.dual_models = self.model.dual_models

        self.reg_lambda = opt.reg_lambda

        if self.opt.lml == 'lwu':
            self.optimizer = utility.make_optimizer(opt, self.model)
            self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        else:  # self.opt.lml == 'ewc':
            self.optimizer = utility.make_base_optimizer(opt, self.model)
            self.dual_optimizers = utility.make_base_dual_optimizer(opt, self.dual_models)

        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8
        self.device = torch.device('cpu' if self.opt.cpu else 'cuda')

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

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        # device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(self.device) for a in args[0]], args[-1].to(self.device)
        return [a.to(self.device) for a in args[0]]

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs

    def ewc_train(self, epocha, train_loader):
        '''训练'''
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        images, labels = None, None
        init_parameters = []
        ewc_loss = 0
        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # ewc 新加的损失
            if (epocha == 1 or batch == 0):
                ewc_loss = 0
                images = lr
                labels = hr
                init_parameters = self.model.model.init_params()
            else:
                fisher_information = self._calc_fisher_information(images, labels)
                ewc_penalty_list = []
                for f, p1, p2 in zip(fisher_information, init_parameters, self.model.model.get_params()):
                    a = (p1 - p2) ** 2 + 5e-4
                    ewc_penalty_list.append(torch.sum(torch.mul(f, a)))
                ewc_loss = torch.stack(ewc_penalty_list).sum()

            # compute primary loss
            loss_primary = self.loss(sr[-1], hr) + ewc_loss
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # compute total loss
            loss = loss_primary + self.opt.dual_weight * loss_dual

            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
                print('batch {}, (Loss: {}, ewc: {})'.format(
                    batch + 1, loss.item(), ewc_loss
                ))
                # self.optimizer.step(self.model.model.reg_params, self.device)
                # for i in range(len(self.dual_optimizers)):
                #     self.dual_optimizers[i].step(self.dual_models[i].reg_params, self.device)
            else:
                print('Skip this batch {}! (Loss: {}, ewc: {})'.format(
                    batch + 1, loss.item(), ewc_loss
                ))

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def si_update_model(self, lr, hr, task, batch):
        unreg_gradients = {}

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        # forward
        sr = self.model(lr[0])

        # Collect the gradients without regularization term
        # compute primary loss
        loss_primary = self.loss(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

        self.optimizer.zero_grad()
        loss_primary.backward(retain_graph=True)

        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        sr = self.model(lr[0])
        sr2lr = []
        for i in range(len(self.dual_models)):
            sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
            sr2lr.append(sr2lr_i)

        # compute primary loss
        loss_primary = self.loss(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])
        # compute dual loss
        loss_dual = self.loss(sr2lr[0], lr[0])
        for i in range(1, len(self.scale)):
            loss_dual += self.loss(sr2lr[i], lr[i])

        # compute total loss
        loss = loss_primary + self.opt.dual_weight * loss_dual

        # Normal update with regularization
        # Calculate the reg_loss only when the regularization_terms exists
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            if (reg_loss > 10 and reg_loss <= 100):
                self.reg_lambda = 1e-1
            elif (reg_loss > 100 and reg_loss <= 1e3):
                self.reg_lambda = 1e-2
            elif (reg_loss > 1e3):
                self.reg_lambda = 1e-5
            loss += self.reg_lambda * reg_loss

        if loss.item() < self.opt.skip_threshold * self.error_last:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].step()
            print('batch {}, (Loss: {}, reg_loss: {})'.format(batch + 1, loss.item(), reg_loss))
        else:
            print('Skip the batch {}, (Loss: {}, reg_loss: {})'.format(batch + 1, loss.item(), reg_loss))

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss, sr, sr2lr

    def si_train(self, task, epocha, train_loader):
        '''训练'''
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            loss, sr, sr2lr = self.si_update_model(lr, hr, task, batch)

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def icarl_train(self, epocha, train_loader):
        '''训练'''
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # icarl loss
            self.criterion_dil = DilLoss(self.scale, lr.size(0))
            dilLoss_primary = self.criterion_dil(sr[-1], hr)

            # compute total loss
            loss = loss_primary + self.opt.dual_weight * loss_dual + self.reg_lambda * dilLoss_primary

            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
                print('batch {}, (Loss: {}, ewc: {})'.format(
                    batch + 1, loss.item(), ewc_loss
                ))
                # self.optimizer.step(self.model.model.reg_params, self.device)
                # for i in range(len(self.dual_optimizers)):
                #     self.dual_optimizers[i].step(self.dual_models[i].reg_params, self.device)
            else:
                print('Skip this batch {}! (Loss: {}, ewc: {})'.format(
                    batch + 1, loss.item(), ewc_loss
                ))

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self, test_loader, epoch, task=0, plot=True):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                tqdm_test = tqdm(test_loader, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        p = utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=test_loader.dataset.benchmark
                        )
                        print('psnr: ', p)
                        eval_psnr += p

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                if plot:
                    self.ckp.log[-1, si] = eval_psnr / len(test_loader)

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[task {}][{} x{}]\tPSNR: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        task,
                        self.opt.data_test, s,
                        eval_psnr / len(test_loader),
                        best[0][si],
                        best[1][si] + 1
                    )
                )

                self.ckp.write_performance(
                    '[epoch {}][task {} x{}]\t PSNR:{:.5f}'.format(
                        epoch,
                        task, s,
                        eval_psnr / len(test_loader)
                    ), file='/performance.txt'
                )
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only and plot:
            self.ckp.save(self, self.scheduler.last_epoch, is_best=(best[1][0] + 1 == epoch))

    def run(self):
        '''单任务训练'''
        self.opt.save = os.path.join(self.opt.save, 'task_' + str(5))
        self.ckp = Checkpoint(self.opt)
        for epoch in range(1, self.opt.epochs + 1):
            self.train(epoch, self.loader_train)
            if epoch % 50 == 0 or epoch == self.opt.epochs:
                self.test(self.loader_test, epoch, task=5, plot=True)

        # 初始化参数
        # self.model.model.init_reg_params()
        # for i in range(len(self.dual_models)):
        #     self.dual_models[i].init_reg_params()
        #
        # while not self.terminate():
        #     self.train(self.scheduler.last_epoch, self.loader_train)
        #     if self.scheduler.last_epoch % 50 == 0 or self.scheduler.last_epoch == self.opt.epochs: #and self.scheduler.last_epoch < self.opt.epochs:
        #         self.test(self.loader_test, self.scheduler.last_epoch)

        # if self.scheduler.last_epoch == self.opt.epochs:
        #     optimizer_ft_model = omega_update(self.model.model.reg_params)
        #     optimizer_ft_duals = []
        #     for i in range(len(self.dual_models)):
        #         optimizer_ft_duals.append(omega_update(self.dual_models[i].reg_params))
        #     # 更新Ω
        #     self.model.model, self.dual_models = compute_omega_grads_norm(self.model.model, self.dual_models,self.loader_train,
        #                                                                   optimizer_ft_model,optimizer_ft_duals,self.device)
        #     self.test(self.loader_test, self.scheduler.last_epoch)

    def mutli_run(self):
        '''多任务训练'''
        num_of_tasks = len(self.loader_train)

        # 训练
        for task in range(4, num_of_tasks + 1):
            self.ckp.write_log(
                'Training the model on task {}\t'.format(task)
            )

            self.opt.save = os.path.join(self.opt.save, 'task_1')
            if task == 1:
                self.opt.save = os.path.join(self.opt.save, 'task_' + str(task))
            else:
                self.opt.save = self.opt.save.replace(os.path.basename(self.opt.save), 'task_' + str(task))
            self.ckp = Checkpoint(self.opt)

            # 把每一个任务对应的数据集取出来
            train_loader = self.loader_train[task - 1]
            test_loader = self.loader_test[task - 1]

            # 初始化正则参数reg_params
            # if task == 1:
            #     # 第一次训练，需要初始化正则参数
            #     self.model.model.init_reg_params()
            #     for i in range(len(self.dual_models)):
            #         self.dual_models[i].init_reg_params()
            # else:
            #     self.model.model.init_reg_params_across_tasks()
            #     for i in range(len(self.dual_models)):
            #         self.dual_models[i].init_reg_params_across_tasks()
            #     print("init across task ++++++++++++++++++++++++")

            # self.test(test_loader, 0, task, plot=False)
            # 实验2：增益训练，每个task都有一个baseline值和(nEpochs-1)个psnr值，除第一个任务外
            for epoch in range(1, self.opt.epochs + 1):
                self.icarl_train(epoch, train_loader)
                if epoch % 50 == 0 or epoch == self.opt.epochs:
                    self.test(test_loader, epoch, task, plot=True)

                # if epoch == self.opt.epochs:
                #     # 创建一个Ω更新的优化器类，传入到模型当中更新参数的Ω值
                #     optimizer_ft_model = omega_update(self.model.model.reg_params)
                #     optimizer_ft_duals = []
                #     for i in range(len(self.dual_models)):
                #         optimizer_ft_duals.append(omega_update(self.dual_models[i].reg_params))
                #     # 更新Ω
                #     self.model.model, self.dual_models = compute_omega_grads_norm(self.model.model, self.dual_models,
                #                                                 train_loader,optimizer_ft_model, optimizer_ft_duals,
                #                                                 self.device)
                #     self.test(test_loader, epoch, task, plot=True)
            # if task > 1:
            #     self.model.model = consolidate_reg_params(self.model.model)
            #     for j in range(len(self.dual_models)):
            #         self.dual_models[j] = consolidate_reg_params(self.dual_models[j])

            # SI 流程
            # self._si_more_opt(train_loader)

        self.ckp.write_log(
            'Computing the forgetting of the model on all task.\t'
        )

    def _calc_fisher_information(self, lr, hr):
        self.optimizer.zero_grad()

        sr = self.model(lr[0])
        mle = self.loss(sr[-1], hr)
        for i in range(1, len(sr)):
            mle += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])
        mle.backward()

        params = []
        for param in self.model.parameters():
            params.append(self.reg_lambda * param.grad ** 2)

        return params

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

    def multi_test(self):
        num_of_tasks = len(self.loader_train)
        epoch = 0
        for task in range(1, num_of_tasks + 1):
            self.ckp.write_log(
                'testing the model on task {}\t'.format(task)
            )

            if task == 1:
                self.opt.save = os.path.join(self.opt.save, 'task_' + str(task))
            else:
                self.opt.save = self.opt.save.replace(os.path.basename(self.opt.save), 'task_' + str(task))
            self.ckp = Checkpoint(self.opt)

            # 把每一个任务对应的数据集取出来
            test_loader = self.loader_test[task - 1]

            self.test(test_loader, epoch, task, plot=False)

    def compute_forgetting(self, task_no, experiment_no, dataloader):
        """
        函数compute_forgetting从存储在performance数列中
        读取先前的性能，并将其与该任务上模型的当前性能进行比较。
        Inputs
        1) task_no: The task number on which you want to compute the forgetting
        2) dataloader: The dataloader that feeds in the data to the model

        Outputs
        1) forgetting: The amount of forgetting undergone by the model

        Function: Computes the "forgetting" that the model has on the
        """

        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                tqdm_test = tqdm(dataloader, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=dataloader.dataset.benchmark
                        )

                    # save test results
                    # if self.opt.save_results:
                    #     self.ckp.save_results_nopostfix(filename, sr, s)

                psnr = eval_psnr / len(dataloader)
                print("task{}, PSNR{:.5f}, beforePSNR{:.5f}".format(task_no, psnr, self.performance[task_no]))
                forgot = psnr - self.performance[task_no]
                self.ckp.write_log(
                    '[FORGETTING][task {}][scale {}]\tPSNR: {:.5f} (Forgetting: {:.5f})'.format(
                        task_no,
                        s,
                        psnr,
                        forgot
                    )
                )

        self.ckp.write_log(
            '[FORGETTING] Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        return forgot