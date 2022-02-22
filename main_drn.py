from DRN_LwU import utility
# 多任务的数据集制作
from DRN_LwU.data import multidata
# 单任务
from DRN_LwU import data
from DRN_LwU import model
from DRN_LwU import loss
from DRN_LwU.option import args
from DRN_LwU.checkpoint import Checkpoint
from DRN_LwU.trainer import Trainer
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loaders = multidata.Data(args)
    # loaders = data.Data(args)
    model = model.Model(args, checkpoint)

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loaders, model, loss, checkpoint)
    # 多任务训练+测试
    t.mutli_run()
    # 单数据训练+测试
    # t.run()
    # 单测试
    # t.test(t.loader_test, t.scheduler.last_epoch)
    checkpoint.done()

