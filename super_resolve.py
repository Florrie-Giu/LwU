from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input', type=str, required=False, default='./dataset/origin_task_MAS/task_001/test_LR_bicubic/X4/0144x4.jpg', help='input image to use')
parser.add_argument('--model', type=str, default='./SRCNN_MAS/newexperiment/multi_adam_srcnn_mas_noresults/task_1/model/srcnn_task_1_150.pth', help='model file to use')
parser.add_argument('--output', type=str, default='./SRCNN_MAS/newexperiment/multi_adam_srcnn_mas_noresults/task_1/144.jpg', help='where to save the output image')
args = parser.parse_args()
print(args)


# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
img = Image.open(args.input)#.convert('RGB')
# y, cb, cr = img.split()


# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
from SRCNN_MAS.model import Net
model = Net(num_channels=3, base_filter=64, upscale_factor=4).to(device)

static = torch.load(args.model, map_location=lambda storage, loc: storage)
model.load_state_dict(static)
# data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
data = (ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
data = data.to(device)

if GPU_IN_USE:
    cudnn.benchmark = True


# ===========================================================
# output and save image
# ===========================================================
out = model(data)
out = out.cpu()
out = out.squeeze(0)
img = ToPILImage()(out)
img.save(args.output)
# out_img_y = out.data[0].numpy()
# out_img_y *= 255.0
# out_img_y = out_img_y.clip(0, 255)
# out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
#
# out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
# out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
# out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

# out_img.save(args.output)

print('output image saved to ', args.output)
