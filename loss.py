#代码主要用于处理和操作稀疏张量数据，计算预测结果与真实结果之间的误差和评估指标#
import torch
import MinkowskiEngine as ME

from data_utils import isin, istopk
criterion = torch.nn.BCEWithLogitsLoss()

def get_bce(data, groud_truth):
# 这段代码定义了一个函数`get_bce(data, groud_truth)`，用于计算输入数据和真实数据之间的二元交叉熵（Binary Cross Entropy，BCE）。具体来说：
# - `mask = isin(data.C, groud_truth.C)`：这一行调用了`isin`函数，输入是`data`和`ground_truth`的坐标，返回一个布尔向量，表示`data`中的每个点是否在`ground_truth`中。
# - `bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))`：这一行计算了`data`和`mask`之间的BCE。`criterion`是一个预定义的损失函数，即二元交叉熵损失函数。`data.F.squeeze()`是`data`的特征向量，`mask.type(data.F.dtype)`将`mask`转换为与`data.F`相同的数据类型。
# - `bce /= torch.log(torch.tensor(2.0)).to(bce.device)`：这一行将BCE除以$\log(2)$，将其转换为以2为底的对数形式，这样计算出的BCE就是以比特（bit）为单位的。
# - `sum_bce = bce * data.shape[0]`：这一行计算了总的BCE，方法是将BCE乘以`data`的数量。
# - `return sum_bce`：这一行返回总的BCE。
# 这个函数可以用于评估点云处理或重建算法的性能，通过计算预测结果和真实结果之间的BCE。
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_metrics(data, groud_truth):
    mask_real = isin(data.C, groud_truth.C)
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    mask_pred = istopk(data, nums, rho=1.0)
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]

def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

