import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader
from pcc_model import PCCModel
from trainer import Trainer

def parse_args():   
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 这行代码创建了一个`argparse.ArgumentParser`对象，它将用于解析命令行参数。`argparse`是Python的一个标准库，用于编写用户友好的命令行接口。
    # `formatter_class=argparse.ArgumentDefaultsHelpFormatter`是一个可选参数，它决定了`argparse`如何显示帮助信息。当设置为`argparse.ArgumentDefaultsHelpFormatter`时，它会在帮助信息中包含每个参数的默认值。
    # 例如，如果你有一个名为`--my_arg`的参数，其默认值为`10`，那么当用户在命令行中输入`python my_script.py --help`时，将会看到类似于以下的输出：
    # --my_arg MY_ARG  (default: 10)
    # 这样，用户就能清楚地知道每个参数的默认值是什么。
    parser.add_argument("--dataset", default='/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/')
    parser.add_argument("--dataset_num", type=int, default=2e4)

    parser.add_argument("--alpha", type=float, default=1., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")
    #parser.add_argument是Python的argparse库中的一个方法，用于向命令行解析器添加参数。这个方法可以接受多个参数来定义如何解析特定的命令行选项。
    # 这段代码定义了一个名为`TrainingConfig`的类，它用于存储训练配置。这个类有以下的属性：
    # - `logdir`：日志目录。如果该目录不存在，将会创建它。
    # - `ckptdir`：检查点目录。如果该目录不存在，将会创建它。
    # - `init_ckpt`：初始检查点。
    # - `alpha`：失真权重。
    # - `beta`：比特率权重。
    # - `lr`：学习率。
    # - `check_time`：记录状态的频率（分钟） 10分钟记录一次。

这个类的实例将在后续的训练过程中被使用，以便于管理和访问这些配置参数。希望这个解释对你有所帮助！如果你有更多的问题，欢迎随时向我提问。😊
    
    
    args = parser.parse_args()
    # 在命令行参数中，-和--是常用的前缀，用来标识参数的名称。一般来说，单个-后面跟的是单字母参数，如-i；而--后面跟的是完整单词或多个单词的参数，如--input。
    # 这种约定有助于区分命令行的参数和其他输入。例如，如果你有一个脚本需要输入文件路径作为参数，你可以这样调用：python my_script.py --input /path/to/my/file。在这里，--input就是一个命令行参数，而/path/to/my/file是这个参数的值。
    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, lr, check_time):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.check_time=check_time
    # 这段代码定义了一个名为TrainingConfig的类，它用于存储训练配置。这个类有以下的属性：
    # logdir：日志目录。如果该目录不存在，将会创建它。
    # ckptdir：检查点目录。如果该目录不存在，将会创建它。
    # init_ckpt：初始检查点。
    # alpha：失真权重。
    # beta：比特率权重。
    # lr：学习率。
    # check_time：记录状态的频率（分钟）。
    # 这个类的实例将在后续的训练过程中被使用，以便于管理和访问这些配置参数。
    # - **日志目录**：这是一个文件夹，用于存储训练过程中生成的日志文件。日志文件通常包含有关模型训练过程的详细信息，如每个训练周期的损失函数值、准确率等。这些信息对于理解模型的训练过程和性能非常有用¹²。
    # - **检查点目录**：这是一个文件夹，用于存储训练过程中的模型检查点。在深度学习中，检查点是在训练过程中的某个时刻保存的模型的状态。检查点通常包含模型的参数和优化器的状态，这样就可以从检查点恢复训练，而不是从头开始。
    # 检查点对于长时间的训练任务非常重要，因为它们可以在训练过程中的任何时刻保存模型的状态，从而防止由于意外（如电源故障或系统崩溃）导致的训练进度丢失¹²。

if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', args.prefix), 
                            #os.path.join('./logs', args.prefix)：这行代码使用os.path.join函数将./logs和args.prefix连接起来，形成日志目录的路径。args.prefix是一个命令行参数，它的值将作为日志目录的一部分。
                            #例如，如果args.prefix的值为'tp'，那么日志目录的路径就会是./logs/tp
                            ckptdir=os.path.join('./ckpts', args.prefix), 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            lr=args.lr, 
                            check_time=args.check_time)
    # model
    model = PCCModel()
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    filedirs = sorted(glob.glob(args.dataset+'*.h5'))[:int(args.dataset_num)]
    # 这行代码执行了以下操作：
    # 1. `glob.glob(args.dataset+'*.h5')`：使用`glob`模块的`glob`函数查找所有以`.h5`结尾的文件。这些文件通常是HDF5格式的文件，常用于存储大量的科学数据。`args.dataset`是数据集的路径，它是通过命令行参数传入的。
    # 2. `sorted(...)`：对找到的文件进行排序。`sorted`函数会按照文件名的字母顺序进行排序¹。
    # 3. `[...][:int(args.dataset_num)]`：这部分代码将排序后的文件列表切片，只保留前`args.dataset_num`个文件。`args.dataset_num`也是通过命令行参数传入的，表示要使用的文件数量。
    # 所以，这行代码的作用是找到数据集路径下所有`.h5`文件，按文件名排序，并只保留前`args.dataset_num`个文件。这些文件的路径被存储在`filedirs`列表中，供后续使用。
    train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
    # 这段代码filedirs[round(len(filedirs)/10):]是Python中的列表切片操作，它用于获取filedirs列表中的一部分元素。
    # len(filedirs)：获取filedirs列表中的元素数量。
    # round(len(filedirs)/10)：将filedirs列表的长度除以10，然后对结果进行四舍五入。这将得到filedirs列表长度的大约10%的位置。
    # filedirs[round(len(filedirs)/10):]：获取filedirs列表中从大约10%的位置开始到最后的所有元素。
    # 例如，如果filedirs列表有100个元素，那么round(len(filedirs)/10)将得到10，filedirs[round(len(filedirs)/10):]将返回一个新的列表，包含filedirs列表中的第10个到第100个元素，总共90个元素。
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
    test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])#前10%数据
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# 这行代码在每个训练周期后更新学习率。学习率在每个周期后减半，但不会低于1×10−5。
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
