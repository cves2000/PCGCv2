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
        #代码的作用是将self.record_set字典中的每一项的值都重置为空列表，以便于下次使用。
        return 

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for _, (coords, feats) in enumerate(tqdm(dataloader)):
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
            torch.cuda.empty_cache()# empty cache.

        self.record(main_tag=main_tag, global_step=self.epoch)

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
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
                # curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce 
                bce_list.append(curr_bce.item())
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            sum_loss = self.config.alpha * bce + self.config.beta * bpp
            # backward & optimize
            sum_loss.backward()
            self.optimizer.step()
            # metric & record
            with torch.no_grad():
                metrics = []
                for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                    metrics.append(get_metrics(out_cls, ground_truth))
                self.record_set['bce'].append(bce.item())
                self.record_set['bces'].append(bce_list)
                self.record_set['bpp'].append(bpp.item())
                self.record_set['sum_loss'].append(bce.item() + bpp.item())
                self.record_set['metrics'].append(metrics)
                if (time.time() - start_time) > self.config.check_time*60:
                    self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                    self.save_model()
                    start_time = time.time()
            torch.cuda.empty_cache()# empty cache.

        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1

        return
