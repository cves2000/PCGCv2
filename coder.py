import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

from pcc_model import PCCModel


class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'
    #__init__方法：初始化函数，接收两个参数，一个是文件名，另一个是熵模型。这个方法将文件名和熵模型保存为类的属性。
    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')#将coords转换为NumPy数组，并将其元素类型转换为整数。
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords)#调用write_ply_ascii_geo函数，将坐标写入一个.ply文件。
        gpcc_encode(self.ply_filename, self.filename+postfix+'_C.bin')#调用gpcc_encode函数，对.ply文件进行编码，生成一个二进制文件。
        os.system('rm '+self.ply_filename)#使用os.system函数执行一个系统命令，删除.ply文件。
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.ply_filename)
        coords = read_ply_ascii_geo(self.ply_filename)#调用read_ply_ascii_geo函数，从.ply文件中读取坐标。
        os.system('rm '+self.ply_filename)#使用os.system函数执行一个系统命令，删除.ply文件。
        
        return coords#返回读取的坐标。


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """
    #这段代码定义了一个名为FeatureCoder的类，该类用于使用学习的熵模型对特征进行编码和解码。
    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):#接收两个参数：feats和postfix，其中postfix的默认值为空字符串
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())#使用熵模型对特征进行压缩，得到压缩后的字符串、最小值和最大值
        shape = feats.shape
        with open(self.filename+postfix+'_F.bin', 'wb') as fout:#打开一个以self.filename+postfix+'_F.bin'命名的二进制文件，准备写入数据，'wb'是文件的打开模式，w表示写入模式，b表示二进制模式。所以，'wb'表示以二进制写入模式打开文件
            fout.write(strings)#将压缩后的字符串写入文件
        with open(self.filename+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())#将特征的形状转换为NumPy数组，然后转换为字节流，写入文件。
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
            
        return 

    def decode(self, postfix=''):
        with open(self.filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*2), dtype=np.int32)#从文件中读取8个字节的内容，将其转换为np.int32类型的NumPy数组，然后将其形状赋值给变量shape
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]#从文件中读取1个字节的内容，将其转换为np.int8类型的NumPy数组，然后将其第一个元素赋值给变量len_min_v。
            min_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]#从文件中读取4*len_min_v个字节的内容，将其转换为np.float32类型的NumPy数组，然后将其第一个元素赋值给变量max_v
            
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])#使用熵模型对字符串进行解压缩，得到特征。
        
        return feats


class Coder():
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder = FeatureCoder(self.filename, model.entropy_bottleneck)

    @torch.no_grad()
    def encode(self, x, postfix=''):
        # Encoder
        y_list = self.model.encoder(x)
        y = sort_spare_tensor(y_list[0])
        num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
        with open(self.filename+postfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        self.feature_coder.encode(y.F, postfix=postfix)
        self.coordinate_coder.encode((y.C//y.tensor_stride[0]).detach().cpu()[:,1:], postfix=postfix)
        
        return y

    @torch.no_grad()
    def decode(self, rho=1, postfix=''):
        # decode coords
        y_C = self.coordinate_coder.decode(postfix=postfix)
        y_C = torch.cat((torch.zeros((len(y_C),1)).int(), torch.tensor(y_C).int()), dim=-1)
        indices_sort = np.argsort(array2vector(y_C, y_C.max()+1))
        y_C = y_C[indices_sort]
        # decode feat
        y_F = self.feature_coder.decode(postfix=postfix)
        y = ME.SparseTensor(features=y_F, coordinates=y_C*8,
                            tensor_stride=8, device=device)
        # decode label
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            num_points[-1] = int(rho * num_points[-1])# update
            num_points = [[num] for num in num_points]
        # decode
        _, out = self.model.decoder(y, nums_list=num_points, ground_truth_list=[None]*3, training=False)

        return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckptdir", default='ckpts/r3_0.10bpp.pth')
    parser.add_argument("--filedir", default='../../../testdata/8iVFB/longdress_vox10_1300.ply')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    args = parser.parse_args()
    filedir = args.filedir

    # load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    outdir = './output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(filedir)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)

    # model
    print('='*10, 'Test', '='*10)
    model = PCCModel().to(device)
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.load_state_dict(ckpt['model'])
    print('load checkpoint from \t', args.ckptdir)

    # coder
    coder = Coder(model=model, filename=filename)

    # down-scale
    if args.scaling_factor!=1: 
        x_in = scale_sparse_tensor(x, factor=args.scaling_factor)
    else: 
        x_in = x

    # encode
    start_time = time.time()
    _ = coder.encode(x_in)
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    x_dec = coder.decode(rho=args.rho)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')

    # up-scale
    if args.scaling_factor!=1: 
        x_dec = scale_sparse_tensor(x_dec, factor=1.0/args.scaling_factor)

    # bitrate
    bits = np.array([os.path.getsize(filename + postfix)*8 \
                            for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
    bpps = (bits/len(x)).round(3)
    print('bits:\t', bits, '\nbpps:\t', bpps)
    print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

    start_time = time.time()
    pc_error_metrics = pc_error(args.filedir, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
