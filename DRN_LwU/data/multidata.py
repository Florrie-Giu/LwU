from importlib import import_module
from torch.utils.data import DataLoader
import os

class Data:
    def __init__(self, args):
        self.loader_train = []
        self.loader_test = []

        data_dir = args.data_dir

        module_train = import_module('DRN_LwU.data.' + args.data_train.lower())
        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109','task_MAS','task_rsi']:
            module_test = import_module('DRN_LwU.data.benchmark')
        else:
            module_test = import_module('data.' +  args.data_test.lower())

        # 制作多任务数据集
        print('===> Loading datasets')

        for tdir in sorted(os.listdir(data_dir)):  # 获取该目录下的子文件
            # 获得目录名
            args.data_dir = os.path.join(data_dir, tdir)
            trainset = getattr(module_train, args.data_train)(args)
            training_data_loader = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )
            # append the dataloaders of these tasks
            self.loader_train.append(training_data_loader)

            # ----------------------------test--------------------
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
            # testset = getattr(module_test, args.data_test)(args, train=False)
            testing_data_loader = DataLoader(
                testset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                pin_memory=not args.cpu
            )
            self.loader_test.append(testing_data_loader)
        print('===> Load finish!')
        # get the number of tasks in the sequence
        print('the number of task are {}' .format(len(self.loader_train)))

