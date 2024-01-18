import os
import numpy as np
import h5py


def read_h5_geo(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:,0:3].astype('int')

    return coords

def write_h5_geo(filedir, coords):
    data = coords.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return
# 这两个函数都是用于处理.h5文件中的3D点云数据。
# read_h5_geo函数：这个函数接收一个文件路径filedir，然后使用h5py.File函数以只读模式打开这个文件。它从文件中读取’data’数据集，并将其转换为NumPy数组。然后，它从数组中提取前三列（即x, y, z坐标），并将其转换为整数类型。最后，它返回这些坐标。
# write_h5_geo函数：这个函数接收一个文件路径filedir和一组坐标coords。它首先将坐标转换为无符号8位整数类型。然后，它使用h5py.File函数以写入模式打开文件，并在其中创建一个名为’data’的数据集，数据集的数据和形状分别为data和data.shape。
# 总的来说，这两个函数提供了一种方便的方法来读取和写入.h5文件中的3D点云数据
# .h5文件是一种用于存储和组织大量数据的文件格式，全称为Hierarchical Data Format version 5（层次数据格式版本5），通常缩写为HDF5或H5。这种文件格式主要用于高性能的读写大量数据，包括但不限于多维数组数据、表格数据等1。
# H5文件是一种开源文件格式，支持大型、复杂的异构数据3。H5使用类似“文件目录”的结构，允许以多种不同的结构化方式组织文件中的数据，就像处理计算机上的文件一样3。H5格式还允许嵌入元数据，使其具有自描述性3。
# 在深度学习的训练中，训练数据和训练后的参数通常会保存为h5格式文件3。对于训练数据来说，深度学习中当训练大量数据时，如果从硬盘中加载再预处理，再传递进网络，这是一个非常耗时的过程。其中从硬盘中读取图片会花费大量时间，更可行在方法是将其存在单个文件中，如h5文件3。因此，h5文件在深度学习训练中具有以下优势3：
# 可以支持大量数据，其中数据集大小大于RAM大小
# 可以增加训练的batch size
# 不用指定数据和数据的shape
# 总的来说，.h5文件是一种非常强大的数据存储格式，它可以有效地处理和存储大量的科学数据
def read_ply_ascii_geo(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('int')

    return coords

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return
# 这两个函数都是用于处理.ply文件中的3D点云数据。
# read_ply_ascii_geo函数：这个函数用于从ASCII格式的.ply文件中读取3D点云数据。它首先打开文件，然后逐行读取文件内容。对于每一行，它会尝试将每个单词转换为浮点数，并将其添加到数据列表中。最后，它会将数据列表转换为NumPy数组，并提取出前三列（即x, y, z坐标）作为坐标返回。
# write_ply_ascii_geo函数：这个函数用于将3D点云数据写入到ASCII格式的.ply文件中。它首先检查文件是否已存在，如果存在则删除。然后，它会打开文件，并写入.ply文件的头部信息。接着，它会遍历坐标，将每个坐标的x, y, z值写入到文件中。最后，它会关闭文件。
# 总的来说，这两个函数提供了一种方便的方法来读取和写入.ply文件中的3D点云数据。
# PLY文件，全称为Polygon File Format或Stanford Triangle Format¹²³⁴⁵，是一种用于存储和描述3D模型的文件格式¹²³⁴⁵。它可以以ASCII或二进制的形式存在¹²³⁴⁵，并且可以存储多边形集合的图形对象¹²³⁴⁵。
# PLY文件格式的目标是提供一种简单且易于实现但通用的格式，足以适用于各种模型¹²³⁴⁵。它被广泛应用于计算机图形学、计算机辅助设计和三维扫描等领域¹²³⁴⁵。
# PLY文件将对象描述为顶点、面和其他元素，以及颜色和法线方向等可以附加到这些元素上的属性¹²³⁴⁵。例如，一个典型的PLY对象定义只是(x,y,z)三元组的顶点列表和面列表，由列表中的索引描述顶点¹²³⁴⁵。大多数PLY文件都包含此核心信息¹²³⁴⁵。
# 总的来说，.ply文件是一种非常强大的数据存储格式，它可以有效地处理和存储大量的科学数据¹²³⁴⁵。
###########################################################################################################

import torch
import MinkowskiEngine as ME

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu() 
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

def sort_spare_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu(), 
                                           sparse_tensor.C.cpu().max()+1))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort], 
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0], 
                                         device=sparse_tensor.device)

    return sparse_tensor_sort

def load_sparse_tensor(filedir, device):
    coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    feats = torch.ones((len(coords),1)).float()
    # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
    
    return x

def scale_sparse_tensor(x, factor):
    coords = (x.C[:,1:]*factor).round().int()
    feats = torch.ones((len(coords),1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)
    
    return x
这段代码主要用于处理3D点云数据。下面是每个函数的详细解释：

# 1. `array2vector(array, step)`: 这个函数将一个2D数组（每个通道都有多个元素）转化为一个1D向量，方法是将每个通道的元素乘以不同的步长然后求和。输入的`array`和`step`都会被转化为长整型并移至CPU进行计算。
# 2. `isin(data, ground_truth)`: 这个函数接收两个形状为[N, D]的torch张量作为输入，返回一个与`data`长度相同的布尔向量，如果`data`的一个元素在`ground_truth`中，则该位置为True，否则为False。
# 3. `istopk(data, nums, rho=1.0)`: 这个函数接收一个稀疏张量`data`和一个形状为[batch_size]的列表`nums`作为输入，返回一个与`data`长度相同的布尔向量，如果`data`的一个元素是前k（=nums*rho）个值，则该位置为True，否则为False。
# 4. `sort_spare_tensor(sparse_tensor)`: 这个函数根据它们的坐标对稀疏张量中的点进行排序。
# 5. `load_sparse_tensor(filedir, device)`: 这个函数从指定的文件目录加载点云数据，并将其转化为稀疏张量。
# 6. `scale_sparse_tensor(x, factor)`: 这个函数将稀疏张量`x`的坐标按照指定的因子进行缩放。
# 这些函数可以用于3D点云数据的预处理、处理和后处理，例如数据加载、数据缩放、数据排序、数据筛选等操作。这些操作在3D点云数据的分析和处理中非常重要。希望这个解释对你有所帮助
