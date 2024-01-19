#管理模型的训练过程
import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME

from loss import get_bce, get_bits, get_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)#调用getlogger方法创建一个日志记录器，并将其保存在self.logger中
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        self.logger.info(model)#模型的信息记录到日志中
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'bce':[], 'bces':[], 'bpp':[],'sum_loss':[], 'metrics':[]}

    def getlogger(self, logdir):#创建一个日志记录器
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self):#加载模型的状态
        """selectively load model
        """
        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)#这行代码使用torch.load函数加载初始检查点
            self.model.load_state_dict(ckpt['model'])#这行代码将加载的模型参数设置到当前模型中
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, #在PyTorch中，self.model.state_dict()是一个方法，它返回一个字典，该字典将每一层的模型参数映射到其对应的参数张量
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return
    # 这段代码定义了Trainer类的save_model方法，它用于保存模型的状态。以下是对这段代码的详细解释：
    # torch.save({'model': self.model.state_dict()}, ...)：这行代码使用torch.save函数保存模型的状态。self.model.state_dict()返回模型的参数字典。
    # os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth')：这部分代码生成了保存模型状态的文件路径。它将检查点目录（self.config.ckptdir）和文件名（‘epoch_’ + 当前训练周期数 + ‘.pth’）连接起来。.pth是PyTorch模型文件的常用扩展名。
    # 这样，我们就可以在训练过程中的任何时刻保存模型的状态，以便于后续的恢复训练或模型评估。
    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})#：这行代码将每个模块的参数和学习率作为一个字典添加到params_lr_list列表中
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer
    # 这行代码创建了一个Adam优化器，用于优化模型的参数。以下是对这段代码的详细解释：
    # - `torch.optim.Adam`：这是PyTorch库中的Adam优化器¹²。Adam是一种常用的优化算法，它结合了Momentum优化和RMSProp优化的思想¹²。
    # - `params_lr_list`：这是一个列表，包含了模型中每个模块的参数和学习率。每个元素是一个字典，其中`"params"`键对应模块的参数，`'lr'`键对应学习率。
    # - `betas=(0.9, 0.999)`：这是Adam优化器的两个超参数，分别对应梯度的移动平均系数和平方梯度的移动平均系数¹²。
    # - `weight_decay=1e-4`：这是权重衰减系数，用于L2正则化。权重衰减可以防止模型过拟合，通过在优化过程中对模型参数施加惩罚，鼓励模型使用更小的权重¹²。
    # 所以，这行代码的作用是创建一个Adam优化器，用于在训练过程中更新模型的参数。

    
    @torch.no_grad()
    # @torch.no_grad()是一个装饰器，用于指定在执行此函数时不需要计算梯度。在PyTorch中，计算梯度是优化模型参数的关键步骤，但在某些情况下，我们并不需要计算梯度。
    # 例如，在评估或测试模型时，我们只关心模型的输出，而不需要更新模型的参数。在这种情况下，我们可以使用@torch.no_grad()装饰器来关闭梯度计算，这样可以节省内存，提高计算速度。
    # 所以，当你看到@torch.no_grad()时，就意味着接下来的操作不会对模型的参数产生任何影响，只是单纯地获取模型的输出。
    def record(self, main_tag, global_step):#用于记录训练过程中的一些关键指标，并将它们打印出来
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)#这行代码的作用是计算record_set字典中每一项的平均值，并将结果存回字典中
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))  
            # 这段代码是在打印self.record_set字典中的每一项。具体来说：
            # for k, v in self.record_set.items():：这行代码遍历self.record_set字典中的每一项，其中k是键，v是值。   
            # np.round(v, 4).tolist()：这行代码将值v（一个NumPy数组）中的每个元素四舍五入到4位小数，并将结果转换为列表。    
            # self.logger.info(k+': '+str(np.round(v, 4).tolist()))：这行代码将键k和经过处理的值转换为字符串，并用冒号:分隔，然后打印出来。    
            # 所以，这段代码的作用是打印出self.record_set字典中每一项的键和值，值是四舍五入到4位小数后的结果。
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  
        #作用是将self.record_set字典中的每一项的值都重置为空列表，以便于下次使用。
        return 

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))#打印
        for _, (coords, feats) in enumerate(tqdm(dataloader)):
        #遍历数据加载器：这行代码遍历数据加载器中的每一个批次。其中，coords和feats分别是每个批次的坐标和特征。_是一个常用的占位符，表示我们不关心这个变量（在这里，它会是批次的索引）。
        #tqdm是一个进度条库，用于显示数据加载的进度。
            # data
            x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
            # # Forward.
            out_set = self.model(x, training=False)
            # loss    
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce 
                bce_list.append(curr_bce.item())
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            sum_loss = self.config.alpha * bce + self.config.beta * bpp
            metrics = []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                metrics.append(get_metrics(out_cls, ground_truth))
            # record
            self.record_set['bce'].append(bce.item())
            self.record_set['bces'].append(bce_list)
            self.record_set['bpp'].append(bpp.item())
            self.record_set['sum_loss'].append(bce.item() + bpp.item())
            self.record_set['metrics'].append(metrics)
            torch.cuda.empty_cache()# empty cache.清空CUDA缓存

        self.record(main_tag=main_tag, global_step=self.epoch)#求平均（bce等参数指标求均值）

        return 

    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))

        start_time = time.time()
        for batch_step, (coords, feats) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            # data
            x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
            # if x.shape[0] > 6e5: continue
            # forward
            out_set = self.model(x, training=True)
            # loss    
            bce, bce_list = 0, []
            # 这段代码是在遍历数据加载器（dataloader）中的每一个批次，并对每一个批次的数据进行处理。以下是对这段代码的详细解释：
            # 遍历数据加载器：for batch_step, (coords, feats) in enumerate(tqdm(dataloader)):这行代码遍历数据加载器中的每一个批次。其中，coords和feats分别是每个批次的坐标和特征。batch_step是批次的索引。tqdm是一个进度条库，用于显示数据加载的进度。        
            # 优化器归零梯度：self.optimizer.zero_grad()这行代码将优化器中所有参数的梯度缓存清零。在PyTorch中，梯度会累积，所以在每个批次开始时，我们需要清零梯度。
            # 数据处理：x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)这行代码将每个批次的坐标和特征转换为一个稀疏张量。这是因为我们的模型是一个3D点云模型，它需要以稀疏张量的形式接收输入数据。
            # 前向传播：out_set = self.model(x, training=True)这行代码将稀疏张量输入到模型中进行前向传播，并获取输出。这里的training=True表示我们当前是在训练模型。
            # 初始化损失：bce, bce_list = 0, []这行代码初始化了二元交叉熵（BCE）的总值和列表。在后续的代码中，我们会计算每个输出类别和真实类别之间的BCE，并将它们添加到这个总值和列表中。
            
            
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            # 在Python中，zip是一个内置函数，它接收一些可迭代对象（如列表、元组等）作为参数，然后将这些可迭代对象中的元素按照顺序配对，形成一个新的迭代器。
            # 例如，如果我们有两个列表a = [1, 2, 3]和b = ['a', 'b', 'c']，那么zip(a, b)会返回一个迭代器，这个迭代器的元素是(1, 'a')、(2, 'b')和(3, 'c')。
                curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
                # curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce 
                bce_list.append(curr_bce.item())
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            sum_loss = self.config.alpha * bce + self.config.beta * bpp
            # backward & optimize
            sum_loss.backward()
            #反向传播：sum_loss.backward()这行代码执行反向传播过程。在PyTorch中，.backward()方法会计算损失函数关于模型参数的梯度。这些梯度用于在接下来的优化步骤中更新模型参数
            self.optimizer.step()
            #优化：self.optimizer.step()这行代码执行优化步骤。在PyTorch中，优化器的.step()方法会根据之前计算的梯度来更新模型参数。这是训练神经网络的关键步骤，因为它使模型能够从数据中学习。
            # metric & record
            with torch.no_grad():
                metrics = []
                for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                    metrics.append(get_metrics(out_cls, ground_truth))
                #这部分代码计算了每个输出类别和真实类别之间的度度，并将这些度量添加到metrics列表中
                self.record_set['bce'].append(bce.item())
                self.record_set['bces'].append(bce_list)
                self.record_set['bpp'].append(bpp.item())
                self.record_set['sum_loss'].append(bce.item() + bpp.item())
                self.record_set['metrics'].append(metrics)
                if (time.time() - start_time) > self.config.check_time*60:
                    self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                    self.save_model()
                    start_time = time.time()
                #检查时间并保存模型：如果训练时间超过了设定的检查时间，那么它会调用record方法来打印这些指标的平均值，并保存当前的模型
            torch.cuda.empty_cache()# empty cache.清空CUDA缓存

        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1
        #记录并保存模型：在每个训练周期结束时，它会调用record方法来打印这些指标的平均值，并保存当前的模型

        return
    # 在PyTorch中，torch.no_grad()是一个上下文管理器，它能够禁止在其作用范围内的代码执行梯度计算。这对于评估或测试模型很有用，因为在这些情况下我们通常只关心模型的输出，而不需要计算梯度或更新模型参数。
    # torch.no_grad()被用在了计算度量和记录关键指标的部分。这是因为这些操作并不需要计算梯度，而且禁止梯度计算可以节省内存，提高计算速度。
