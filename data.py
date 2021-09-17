import glob
import os
import torch
import os.path as osp
from PIL import Image

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class ImageTransform:
    def __init__(self, img_size=256):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])}

    def __call__(self, img, phase='train'):
        return self.transform[phase](img)


class FaceDataPaired(Dataset):
    def __init__(self, root_dir_x, root_dir_y):
        self.transform = ImageTransform()

        images_x = glob.glob(osp.join(root_dir_x, '*', '*'))
        images_y = glob.glob(osp.join(root_dir_y, '*', '*'))
        if len(images_x) < len(images_y):
            self.images = images_x
            self.dir0 = root_dir_x
            self.dir1 = root_dir_y
        else:
            self.images = images_y
            self.dir0 = root_dir_y
            self.dir1 = root_dir_x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sep_path = self.images[index].split(os.sep)
        name0 = os.sep.join(sep_path[-2:])
        name1 = name0.replace('_Mask.jpg', '.png')
        path0 = osp.join(self.dir0, name0)
        path1 = osp.join(self.dir1, name1)

        img0 = Image.open(path0).convert('RGB')
        img1 = Image.open(path1).convert('RGB')

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1


class FaceDataUnpaired(Dataset):
    def __init__(self, root_dir_x, root_dir_y):
        self.transform = ImageTransform()

        images_x = glob.glob(osp.join(root_dir_x, '*', '*'))
        images_y = glob.glob(osp.join(root_dir_y, '*', '*'))
        if len(images_x) < len(images_y):
            self.images0 = images_x
            self.images1 = images_y
        else:
            self.images0 = images_y
            self.images1 = images_x

    def __len__(self):
        return len(self.images0) // 2

    def __getitem__(self, index):
        img0 = Image.open(self.images0[index]).convert('RGB')
        img1 = Image.open(self.images1[index + len(self)]).convert('RGB')

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        return img0, img1


class FaceDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir_x='/var/home/menno/Documents/huawei/data/train_data/FFHQ',
                 train_dir_y='/var/home/menno/Documents/huawei/data/train_data/CMFD',
                 test_dir_x='/var/home/menno/Documents/huawei/data/test_data/FFHQ',
                 test_dir_y='/var/home/menno/Documents/huawei/data/test_data/CMFD',
                 batch_size=4, num_workers=8):
        super().__init__()
        self.train_dir_x = train_dir_x
        self.train_dir_y = train_dir_y
        self.test_dir_x = test_dir_x
        self.test_dir_y = test_dir_y
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        train_data = FaceDataUnpaired(self.train_dir_x, self.train_dir_y)
        test_data = FaceDataPaired(self.test_dir_x, self.test_dir_y)

        self.train = train_data
        self.val, self.test = random_split(
            test_data, [1000, 8608],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


def test():
    dm = FaceDataModule()
    dm.setup(None)
    for img in dm.train_dataloader():
        print('Shapes:', img[0].shape, img[1].shape)
        print('Difference:', img[0].sub(img[1]).sum())
        break

    for img in dm.val_dataloader():
        print('Shapes:', img[0].shape, img[1].shape)
        print('Difference:', img[0].sub(img[1]).sum())
        break


if __name__ == "__main__":
    test()
