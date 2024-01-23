# RoundNoGradient：这个类的目的是创建一个特殊的四舍五入操作，这个操作在反向传播时不会产生梯度。也就是说，这个操作不会影响模型的学习过程。
# Low_bound：这个类的目的是确保输入的值不会低于一个特定的阈值（在这里是1e-9）。这是为了防止在计算过程中出现数值问题。
# EntropyBottleneck：这个类是一个特殊的神经网络层，它的目的是在压缩和解压缩过程中，尽可能地保留信息，同时减少所需的存储空间。
# import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

import torchac


class RoundNoGradient(torch.autograd.Function):
    """ TODO: check. """
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g
# 这段代码定义了一个名为RoundNoGradient的自定义PyTorch函数。这个函数的主要目的是实现一个无梯度的四舍五入操作。
# forward方法：在前向传播过程中，它接收一个输入x，并返回x的四舍五入结果。
# backward方法：在反向传播过程中，它接收一个梯度g，并直接返回这个梯度。这意味着在计算梯度时，它不会对梯度进行任何修改，也就是说，这个四舍五入操作是无梯度的。
# 这种无梯度的四舍五入操作在某些情况下是有用的，例如，在进行量化操作时，我们可能希望量化步骤不影响反向传播的梯度计算。

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):#不加self：在Python中，被@staticmethod装饰的方法被称为静态方法，它们可以直接通过类来调用，而不需要创建类的实例。因此，静态方法不需要self参数。
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        # 在Python中，当你在一个变量后面加上逗号,，这实际上是创建了一个元组。元组是Python的一种数据结构，它可以包含多个元素。
        # 在这段代码中：
        # x, = ctx.saved_tensors
        # AI 生成的代码。仔细查看和使用。 有关常见问题解答的详细信息.
        # ctx.saved_tensors返回的是一个元组，元组中包含了在forward方法中保存的所有张量。通过在x后面加上逗号,，我们实际上是在进行元组解包，也就是将元组中的元素分别赋值给变量。
        # 如果ctx.saved_tensors只包含一个元素，那么x, = ctx.saved_tensors就等价于x = ctx.saved_tensors[0]
        grad1 = g.clone()
        try:
            grad1[x<1e-9] = 0#这行代码将grad1中对应x小于1e-9的位置的梯度设置为0。这样做的目的是为了避免在这些位置出现数值稳定性问题。
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy()<0.0)
        # 这行代码的主要作用是创建一个布尔数组pass_through_if，它的值为True当且仅当x大于等于1e-9或g小于0.0
        # 这段代码中包含了几个操作，让我们逐一解释：
        # x.cpu()：这个操作将x（一个PyTorch张量）从当前设备（如果它在GPU上）移动到CPU。 
        # detach()：这个操作创建了一个新的张量，新的张量与原始张量共享数据，但不会在反向传播中计算梯度。这通常在我们只需要张量的数据，但不需要计算梯度时使用。      
        # numpy()：这个操作将PyTorch张量转换为NumPy数组。NumPy是Python的一个库，提供了大量的数学函数和多维数组对象。       
        # np.logical_or(...)：这是NumPy的一个函数，用于计算输入数组中对应元素的逻辑或（OR）操作。也就是说，如果任一输入数组中的对应元素为True，结果数组中的对应元素就为True。    
        # 所以，np.logical_or(x.cpu().detach().numpy() >= 1e-9, ...)这段代码的意思是，首先将x移动到CPU，然后创建一个不需要梯度的副本，并将其转换为NumPy数组，然后检查数组中的每个元素是否大于等于1e-9，结果是一个布尔数组。
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t  #grad1和t张量的元素级乘积（哈达玛积），这是最终的梯度。
    # 关于@staticmethod：
    # 如果你不使用`@staticmethod`装饰器，那么`forward`和`backward`方法就会变成实例方法，你需要先创建`Low_bound`类的一个实例，然后通过这个实例来调用这些方法。这是一个例子：
    # # 创建Low_bound类的一个实例
    # low_bound_instance = Low_bound()
    # # 通过实例调用forward方法
    # result = low_bound_instance.forward(ctx, x)
    # 在这个例子中，我们首先创建了`Low_bound`类的一个实例`low_bound_instance`，然后我们通过这个实例来调用`forward`方法。
    # 但是，请注意，对于PyTorch自定义函数，我们通常会使用静态方法，因为这样可以直接通过类名来调用这些方法，而不需要创建类的实例。这也是为什么在你给出的代码中，`forward`和`backward`方法被声明为静态方法的原因。

class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""
    
    def __init__(self, channels, init_scale=8, filters=(3,3,3)):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            #
            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias)# copy or fill?
            self._biases.append(self.bias)
            #       
            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            self._factors.append(self.factor)

    def _logits_cumulative(self, inputs):
        """Evaluate logits of the cumulative densities.
        
        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)
        
        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def _likelihood(self, inputs):
        """Estimate the likelihood.
        inputs shape: [points, channels]
        """
        # reshape to (channels, 1, points)
        inputs = inputs.permute(1, 0).contiguous()# [channels, points]
        shape = inputs.size()# [channels, points]
        inputs = inputs.view(shape[0], 1, -1)# [channels, 1, points]
        inputs = inputs.to(self.matrix.device)
        # Evaluate densities.
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        # reshape to (points, channels)
        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0)

        return likelihood

    def forward(self, inputs, quantize_mode="noise"):
        """Pass a tensor through the bottleneck.
        """
        if quantize_mode is None: outputs = inputs
        else: outputs = self._quantize(inputs, mode=quantize_mode)
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad()
    def compress(self, inputs):
        # quantize
        values = self._quantize(inputs, mode="symbols")
        # get symbols
        min_v = values.min().detach().float()
        max_v = values.max().detach().float()
        symbols = torch.arange(min_v, max_v+1)
        symbols = symbols.reshape(-1,1).repeat(1, values.shape[-1])# (num_symbols, channels)
        # get normalized values
        values_norm = values - min_v
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        values_norm = values_norm.to(torch.int16)

        # get pmf
        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1,0)# (channels, num_symbols)

        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic encoding
        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True)

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape, channels):
        # get symbols
        symbols = torch.arange(min_v, max_v+1)
        symbols = symbols.reshape(-1,1).repeat(1, channels)

        # get pmf
        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1,0)
        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic decoding
        out_cdf = cdf.unsqueeze(0).repeat(shape[0], 1, 1).detach().cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v

        return values
    # EntropyBottleneck是一个自定义的PyTorch模块，它实现了一个灵活的概率密度模型，用于估计其输入张量的熵。这个模块的设计思想来源于论文"Variational image compression with a scale hyperprior"。
    # 在图像和视频压缩中，EntropyBottleneck层通常用于对网络的中间表示进行熵编码，以实现更高效的压缩。它的主要功能包括：    
    # 量化：在前向传播过程中，EntropyBottleneck会对输入进行量化，这是压缩过程的一个重要步骤。 
    # 估计可能性：EntropyBottleneck会估计量化后的值的可能性，这对于熵编码是必要的。
    # 压缩和解压缩：EntropyBottleneck提供了compress和decompress方法，用于对输入进行压缩和解压缩。在压缩过程中，它会对输入进行量化，然后计算概率质量函数（pmf）和累积分布函数（cdf），最后进行算术编码。在解压缩过程中，它会进行算术解码，然后将解码后的值反量化。
    # 总的来说，EntropyBottleneck是实现高效图像和视频压缩的关键组件。
