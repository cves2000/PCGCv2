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

def get_bits(likelihood):#计算likelihood的负对数似然的总和，结果以比特为单位
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_metrics(data, groud_truth):#data groud_truth稀疏张量
    # 稀疏张量的主要属性包括：
    # C：一个二维张量，表示非零元素的坐标。
    # F：一个一维张量，表示非零元素的特征。
    mask_real = isin(data.C, groud_truth.C)#.C即坐标
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    #nums是一个一维列表，其中每个元素表示ground_truth中对应分解坐标的长度。具体来说，nums列表的长度等于ground_truth中分解坐标的数量，nums列表中的每个元素是对应分解坐标的长度。
    mask_pred = istopk(data, nums, rho=1.0)
    #这一行调用了istopk函数，输入是data、nums和rho，返回一个布尔向量，表示data中的每个点是否在前k个点中（对特征值进行排序 取前k个）个人理解：特征值越大意味着点云存在概率越大，通过将特征值较大的点云索引设置为true，来表示该点的预测为真
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]
    # 稀疏张量的例子。假设我们有一个3D点云，其中包含以下三个点：
    # - 点1：坐标为(0, 0, 0)，特征为1.0
    # - 点2：坐标为(1, 2, 3)，特征为2.0
    # - 点3：坐标为(4, 5, 6)，特征为3.0
    # 我们可以将这个点云表示为一个稀疏张量，如下所示：
    # data = SparseTensor(
    #     coordinates=torch.tensor([[0, 0, 0], [1, 2, 3], [4, 5, 6]]),
    #     features=torch.tensor([1.0, 2.0, 3.0])
    # )
    # 在这个例子中，`data.C`是一个二维张量，表示非零元素的坐标，即`[[0, 0, 0], [1, 2, 3], [4, 5, 6]]`。`data.F`是一个一维张量，表示非零元素的特征，即`[1.0, 2.0, 3.0]`。
    
def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

    # TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]：这一行计算了真正例（True Positive，TP）的数量，即预测为正且实际为正的样本数量。
    # FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]：这一行计算了假负例（False Negative，FN）的数量，即预测为负但实际为正的样本数量。
    # FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]：这一行计算了假正例（False Positive，FP）的数量，即预测为正但实际为负的样本数量。
    # TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]：这一行计算了真负例（True Negative，TN）的数量，即预测为负且实际为负的样本数量。
    # precision = TP / (TP + FP + 1e-7)：这一行计算了精确度（Precision），即预测为正的样本中实际为正的比例。
    # recall = TP / (TP + FN + 1e-7)：这一行计算了召回率（Recall），即实际为正的样本中预测为正的比例。
    # IoU = TP / (TP + FP + FN + 1e-7)：这一行计算了交并比（Intersection over Union，IoU），即预测为正和实际为正的样本的交集与并集的比例。
    # return [round(precision, 4), round(recall, 4), round(IoU, 4)]：这一行返回计算出的精确度、召回率和IoU，保留四位小数。
    #这个函数可以用于评估分类模型的性能，通过计算预测结果和真实结果之间的各种评估指标
