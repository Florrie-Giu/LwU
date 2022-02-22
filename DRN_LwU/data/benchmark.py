import os
from DRN_LwU.data import srdata


class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, data_dir):
        # data_dir = '../self_adapation/dataset/task_MAS/task_003/'
        print('mas benchmark path', data_dir)
        self.dir_hr = os.path.join(data_dir, 'test')
        self.dir_lr = os.path.join(data_dir, 'test_LR_bicubic')
        temp_ext = os.path.splitext(os.listdir(self.dir_hr)[0])[1]
        self.ext = (temp_ext, temp_ext)