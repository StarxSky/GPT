import os
import PIL
import torch
import numpy as np


from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, pt_dataset, Group, perm=None):
        self.pt_dataset = pt_dataset  # 设置图片数据源,类型为列表
        self.Group = Group # 设置集群
        self.perm = torch.arange(32 * 32) if perm is None else perm
        self.vocab_size = Group.size(0)
        self.block_size = 32 * 32 - 1

    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        # 如果数据集pt_dataset中含有标签则应该在x后面加上y值来进行解包否则会报错unpack
        x  = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, 3)  # flatten out all pixels
        x = x[self.perm].float()  # 使用任何固定排列和重新shuffle像素值
        a = ((x[:, None, :] - self.Group[None, :, :]) ** 2).sum(-1).argmin(1)  # cluster assignments
        return a[:-1], a[1:]  # 一直预测下一个序列




class Images_load_local(Dataset):
    def __init__(self, pt_dataset_path) :
        self.pt_datas = pt_dataset_path
        self.im_name = os.listdir(pt_dataset_path)

    def __len__(self):
        return len(self.im_name)

    def __getitem__(self, idx):
       image_path = os.path.join(self.pt_datas,self.im_name[idx])
       image = PIL.Image.open(image_path)
       image = image.resize([32,32])
       image = torch.from_numpy(np.array(image))
       # ===============TEST====================
       #image_transforms = transforms.Compose([transforms.Grayscale(1)])
       #image = image_transforms(image)
       return image



