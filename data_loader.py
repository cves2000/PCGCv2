import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_utils import read_h5_geo, read_ply_ascii_geo

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    #InfSampler类能够抽取样本，是因为它内部维护了一个排列（permutation）列表，这个列表的长度等于数据集的样本总数。
    #每次调用__next__方法时，它都会从这个列表中弹出（pop）一个元素，这个元素就是下一个要抽取的样本的索引。
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()
    #__init__函数：初始化函数，接收两个参数，data_source是需要采样的数据集，shuffle决定是否打乱样本的顺序。
     #reset_permutation函数：重置排列函数，如果shuffle=True，则生成一个随机排列；如果shuffle=False，则生成一个顺序排列
    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()
    # perm = len(self.data_source)：获取数据集的长度，即样本的总数。
    # if self.shuffle: perm = torch.randperm(perm)：如果self.shuffle=True，则生成一个随机排列；如果self.shuffle=False，则perm保持为数据集的长度。
    # self._perm = perm.tolist()：将排列转换为列表，并保存到self._perm中。
    def __iter__(self):
        return self
    #__iter__函数：返回迭代器自身，使得该类的实例可以直接用于迭代。
    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()
    #__next__函数：返回下一个样本的索引。如果所有的样本都已经被抽取过，那么就调用reset_permutation函数重新开始。
    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
# 这是一个名为collate_pointcloud_fn的函数，它的主要作用是将多个样本整理成一个批次的数据。具体来说，它的功能如下：
# 移除空样本：首先，这个函数会遍历输入的数据列表list_data，并将其中的非空样本添加到新的列表new_list_data中。同时，它还会计算被移除的空样本的数量num_removed。
# 检查数据列表：如果新的数据列表new_list_data为空，即所有的样本都是空的，那么这个函数会抛出一个值错误（ValueError）。
# 解压数据：然后，这个函数会使用zip函数将数据列表中的坐标和特征分别解压到coords和feats中。
# 整理数据：最后，这个函数会使用ME.utils.sparse_collate函数将解压的坐标和特征转换为批次数据，得到coords_batch和feats_batch。
# 总的来说，collate_pointcloud_fn函数是一个数据整理函数，它可以将多个样本整理成一个批次的数据，方便后续的计算和处理。
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return coords_batch, feats_batch


class PCDataset(torch.utils.data.Dataset):
# 这是一个名为PCDataset的类，它继承自PyTorch的Dataset类。这个类的主要作用是处理和加载3D点云数据。
# 总的来说，PCDataset类是一个用于处理和加载3D点云数据的工具，它可以帮助用户更方便地在PyTorch中使用3D点云数据。
    def __init__(self, files):
    # __init__函数：初始化函数，接收一个文件列表files作为参数，并初始化一个空的缓存cache和一个缓存百分比last_cache_percent。
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files
    def __len__(self):
    # __len__函数：返回数据集的长度，即文件列表的长度。
        return len(self.files)
    def __getitem__(self, idx):
    # __getitem__函数：根据索引idx获取一个样本。首先，它会检查该样本是否已经在缓存中，如果是，则直接从缓存中获取；如果不是，则读取文件并将其添加到缓存中。
    #然后，它会计算当前的缓存百分比，如果缓存百分比每增加10%，就更新last_cache_percent。最后，它会将特征转换为浮点类型，并返回坐标和特征。
    #总的来说，__getitem__方法是用于根据索引获取样本的函数，它可以从文件中读取数据，也可以从缓存中获取数据，以提高数据获取的效率。
        filedir = self.files[idx]

        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            if filedir.endswith('.h5'): coords = read_h5_geo(filedir)
            if filedir.endswith('.ply'): coords = read_ply_ascii_geo(filedir)
            feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            # cache
            self.cache[idx] = (coords, feats)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        feats = feats.astype("float32")
        return (coords, feats)
    
def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
# 这是一个名为make_data_loader的函数，它的主要作用是创建一个数据加载器。具体来说，它的功能如下：
# 设置参数：首先，这个函数会创建一个参数字典args，包括批次大小batch_size、工作进程数num_workers、数据整理函数collate_fn、是否在内存中固定数据pin_memory、以及是否丢弃最后一个不完整的批次drop_last。
# 设置采样器：然后，如果repeat=True，那么这个函数会创建一个InfSampler实例作为采样器；如果repeat=False，那么它会直接使用shuffle参数。
# 创建数据加载器：最后，这个函数会使用torch.utils.data.DataLoader类创建一个数据加载器，并返回它。
# 总的来说，make_data_loader函数是一个用于创建数据加载器的工具，它可以帮助用户更方便地从数据集中加载数据                    
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)#infsampler本代码第一个类
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)
    #torch.utils.data.DataLoader是PyTorch提供的一个工具，它可以自动地将数据集分割成多个批次，方便进行批次训练。
    #**args是Python的语法，它表示将字典args中的键值对作为关键字参数传递给DataLoader函数。
    #例如，如果args是{'batch_size': 1, 'shuffle': True}，那么DataLoader(dataset, **args)就等价于DataLoader(dataset, batch_size=1, shuffle=True)。              
    return loader


if __name__ == "__main__":
#这段代码是在Python的主程序中执行的部分，也就是当你直接运行这个Python文件时，会执行if __name__ == "__main__":后面的代码。具体来说，它的功能如下：
# 获取文件列表：使用glob.glob函数获取指定目录下的所有.ply文件，并按照字母顺序排序。
# 创建数据集：使用PCDataset类创建一个数据集，这个数据集包含了文件列表中的前10个文件。
# 创建数据加载器：使用make_data_loader函数创建一个数据加载器，这个数据加载器可以从数据集中按照指定的参数加载数据。
# 遍历数据加载器：使用一个循环来遍历数据加载器，并打印出每个批次的坐标和特征。
# 创建数据加载器的迭代器：使用iter函数创建一个数据加载器的迭代器。
# 使用迭代器加载数据：使用一个循环来通过迭代器加载数据，并打印出每个批次的坐标和特征。
# 总的来说，这段代码的主要作用是展示如何使用PCDataset类和make_data_loader函数来处理和加载3D点云数据。
    # filedirs = sorted(glob.glob('/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/'+'*.h5'))
    filedirs = sorted(glob.glob('/home/ubuntu/HardDisk1/point_cloud_testing_datasets/8i_voxeilzaed_full_bodies/8i/longdress/Ply/'+'*.ply'))
    test_dataset = PCDataset(filedirs[:10])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False,
                                        collate_fn=collate_pointcloud_fn)
    for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
        print("="*20, "check dataset", "="*20, 
            "\ncoords:\n", coords, "\nfeat:\n", feats)

    test_iter = iter(test_dataloader)
    print(test_iter)
    for i in tqdm(range(10)):
        coords, feats = test_iter.next()
        print("="*20, "check dataset", "="*20, 
            "\ncoords:\n", coords, "\nfeat:\n", feats)



