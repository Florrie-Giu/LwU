from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img_YCbCr(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def load_img(filepath):
    # RGB
    img = Image.open(filepath)
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, split='train', input_transform=None, target_transform=None, old_output_map={}):
        super(DatasetFromFolder, self).__init__()
        if split == 'train':
            # self.lr_dir = join(image_dir, 'train_LR_bicubic', 'X4')
            self.hr_dir = join(image_dir, 'train')
        else:

            self.hr_dir = join(image_dir, 'test')
        self.image_filenames = [join(self.hr_dir, x) for x in listdir(self.hr_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        #Lwf: more task
        self.obtain_old_outputs = False
        self.old_output_map = old_output_map

    def __getitem__(self, index):
        filename = os.path.basename(self.image_filenames[index])
        input_image = load_img(self.image_filenames[index])
        # target = load_img(join(self.hr_dir, filename.replace('x4','')))
        target = input_image.copy()
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        # Use for training of LwF
        if len(self.old_output_map):
            old_output = self.old_output_map[filename]
            return input_image, target, old_output

            # Use for get name - old task output mapping
        if self.obtain_old_outputs:
            return input_image, target, filename

        return input_image, target, filename

    def __len__(self):
        return len(self.image_filenames)
