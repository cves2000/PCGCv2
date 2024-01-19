import torch
import MinkowskiEngine as ME

from autoencoder import Encoder, Decoder
from entropy_model import EntropyBottleneck


class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood
    # 这段代码定义了get_likelihood函数，它是PCCModel类的一个方法。这个函数的主要作用是获取数据的似然性。下面是对这段代码的详细解释：
    # data_F, likelihood = self.entropy_bottleneck(data.F, quantize_mode=quantize_mode)：这行代码调用了EntropyBottleneck类的实例self.entropy_bottleneck，输入是数据的特征data.F和量化模式quantize_mode。EntropyBottleneck是一个用于实现熵编码的类，它可以对输入的特征进行量化，并返回量化后的特征data_F和对应的似然性likelihood。
    # data_Q = ME.SparseTensor(...)：这行代码创建了一个ME.SparseTensor的实例data_Q。ME.SparseTensor是MinkowskiEngine库中的一个类，用于表示稀疏张量。它的输入包括量化后的特征data_F，坐标映射键coordinate_map_key，坐标管理器coordinate_manager，以及设备device。
    # 最后，这个函数返回量化后的数据data_Q和对应的似然性likelihood。
    # 这段代码的主要作用是实现一个3D点云的压缩和解压模型。简单来说，就是将3D点云数据（比如一个物体的3D扫描结果）进行压缩，然后再解压恢复出原始的3D点云数据。
    # 具体来说，这个模型包含三个部分：编码器、解码器和熵瓶颈。
    # 编码器：它的作用是将输入的3D点云数据进行编码，也就是将数据转换成一种更便于压缩和传输的形式。
    # 熵瓶颈：这是一个用于实现熵编码的部分，它可以对编码后的数据进行量化，并计算出数据的似然性。似然性是一个表示数据出现概率的值，这个值越大，表示数据出现的可能性越大。
    # 解码器：它的作用是将量化后的数据进行解码，恢复出原始的3D点云数据。
    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        #在这段代码中，ground_truth_list = y_list[1:] + [x]的作用是将编码器的输出（除了第一个元素）和输入数据x合并到一起，形成一个新的列表ground_truth_list。
        # 这里的加号+是Python中的列表连接操作，它可以将两个列表合并成一个新的列表。y_list[1:]表示取y_list的第二个元素到最后一个元素，[x]则是将输入数据x转换成一个只有一个元素的列表。这两个列表通过加号+连接在一起，形成了新的列表ground_truth_list。
        # 这样做的目的是为了在后续的解码过程中，解码器可以同时参考编码器的输出和原始的输入数据，以便更准确地进行解码。
        # y_list列表末尾添加元素x
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]
            # 这段代码是在计算ground_truth_list中每个元素的坐标数量，并将结果保存在nums_list中。
            # 具体来说，ground_truth.decomposed_coordinates是一个列表，它包含了ground_truth（即需要解码的数据）的所有坐标。len(C)则是计算每个坐标集合C的长度，也就是坐标的数量。     
            # 这样，nums_list就是一个二维列表，它的每个元素是一个列表，表示对应的ground_truth的坐标数量。
            # 这个nums_list在后续的解码过程中会被用到，它可以帮助解码器知道每个ground_truth的坐标数量，从而更准确地进行解码。
            #关于这段代码的例子。
            # ground_truth_list = [
            #     {"decomposed_coordinates": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
            #     {"decomposed_coordinates": [[10, 11, 12], [13, 14, 15]]}
            # ]
            # 在这个例子中，`ground_truth_list`是一个列表，包含了两个字典。每个字典都有一个键`"decomposed_coordinates"`，对应的值是一个列表，包含了一组坐标。
            # 当我们执行`nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]`这行代码后，我们得到的`nums_list`是：
            # nums_list = [[3, 3, 3], [3, 3]]
            # 可以看到，`nums_list`包含了`ground_truth_list`中每个元素的坐标数量。每个元素是一个列表，表示对应的`ground_truth`的坐标数量。

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y, 
            quantize_mode="noise" if training else "symbols")

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)
        #这段代码是在调用解码器进行解码操作。具体来说，解码器接收四个参数：y_q、nums_list、ground_truth_list和training。
        # y_q：这是量化后的数据，它是由熵瓶颈模型对编码后的数据进行量化得到的。
        # nums_list：这是一个列表，包含了ground_truth_list中每个元素的坐标数量。
        # ground_truth_list：这是一个列表，包含了所有需要解码的数据。这个列表是由编码器的输出（除了第一个元素）和输入数据x组成的。
        # training：这是一个布尔值，表示当前是否处于训练模式。
        # 解码器的输出是两个元素：out_cls_list和out。
        # out_cls_list：这是一个列表，包含了解码后的分类结果。
        # out：这是解码后的数据，也就是恢复出的原始3D点云数据。
        return {'out':out,
                'out_cls_list':out_cls_list,
                'prior':y_q, 
                'likelihood':likelihood, 
                'ground_truth_list':ground_truth_list}
        #这段代码定义了`forward`函数，它是`PCCModel`类的一个方法。这个函数的主要作用是实现模型的前向传播过程。下面是对这段代码的详细解释：
        # 1. **编码器**：首先，输入数据`x`被送入编码器进行编码，得到的结果是一个列表`y_list`。列表的第一个元素`y`是编码后的数据，剩余的元素和输入数据`x`一起构成了`ground_truth_list`，这个列表包含了所有需要解码的数据。然后，计算每个需要解码的数据的坐标数量，结果保存在`nums_list`中。
        # 2. **量化器和熵模型**：接着，编码后的数据`y`被送入`get_likelihood`函数进行量化，得到的结果是量化后的数据`y_q`和对应的似然性`likelihood`。如果当前是训练模式，那么量化模式设置为"noise"，否则设置为"symbols"。
        # 3. **解码器**：最后，量化后的数据`y_q`被送入解码器进行解码，得到的结果是解码后的数据`out`和对应的分类结果列表`out_cls_list`。
        # 这个函数的返回值是一个字典，包含了解码后的数据`out`、分类结果列表`out_cls_list`、量化后的数据`y_q`、似然性`likelihood`和需要解码的数据列表`ground_truth_list`。
if __name__ == '__main__':
    model = PCCModel()
    print(model)

