import torch
import numpy as np
import os
from pcc_model import PCCModel
from coder import Coder
import time
from data_utils import load_sparse_tensor, sort_spare_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from pc_error import pc_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(filedir, ckptdir_list, outdir, resultdir, scaling_factor=1.0, rho=1.0, res=1024):
    # load data这是一个名为test的函数，接收多个参数，包括文件目录、检查点目录列表、输出目录、结果目录、缩放因子、rho和分辨率。
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)#加载稀疏的3D点云数据。
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')
    # x = sort_spare_tensor(input_data)

    # output filename
    if not os.path.exists(outdir): os.makedirs(outdir)#如果输出目录不存在，则创建它。
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])#构建输出文件名。
    print('output filename:\t', filename)
    
    # load model
    model = PCCModel().to(device)创建PCC模型并将其移动到指定的设备（CPU或GPU）上。

    for idx, ckptdir in enumerate(ckptdir_list):#遍历检查点目录列表。
        print('='*10, idx+1, '='*10)
        # load checkpoints
        assert os.path.exists(ckptdir)#确保检查点目录存在。
        ckpt = torch.load(ckptdir)#加载检查点。
        model.load_state_dict(ckpt['model'])#将模型的状态从检查点中加载。
        print('load checkpoint from \t', ckptdir)
        coder = Coder(model=model, filename=filename)#创建一个Coder实例，用于编码和解码。

        # postfix: rate index
        postfix_idx = '_r'+str(idx+1)#构建后缀，用于区分不同的检查点。

        # down-scale
        if scaling_factor!=1: 
            x_in = scale_sparse_tensor(x, factor=scaling_factor)#如果指定了缩放因子，则对输入数据进行缩放。
        else: 
            x_in = x#否则，使用原始输入数据。

        # encode
        start_time = time.time()
        _ = coder.encode(x_in, postfix=postfix_idx)#对输入数据进行编码。
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)#将编码时间保存到变量 time_enc 中。

        # decode
        start_time = time.time()
        x_dec = coder.decode(postfix=postfix_idx, rho=rho)#对编码后的数据进行解码。
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)#将解码时间保存到变量 time_dec 中。然后是还原（如果有缩放因子的话）：

        # up-scale
        if scaling_factor!=1: 
            x_dec = scale_sparse_tensor(x_dec, factor=1.0/scaling_factor)#如果使用了缩放因子，将解码后的数据还原到原始分辨率。

        # bitrate
        bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])#计算各个文件的比特数。
        bpps = (bits/len(x)).round(3)#计算每个点的平均比特数。
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))#打印总比特数和平均比特数。

        # distortion
        start_time = time.time()
        write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])#将解码后的点云写入PLY文件。
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(filedir, filename+postfix_idx+'_dec.ply', 
                                    res=res, normal=True, show=False)#计算点云误差度量。
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])#打印D1 PSNR值

        # save results
        results = pc_error_metrics #将点云误差度量结果赋值给变量 results。
        results["num_points(input)"] = len(x)#记录输入点云的点数。
        results["num_points(output)"] = len(x_dec)#记录解码后点云的点数。
        results["resolution"] = res#记录分辨率。
        results["bits"] = sum(bits).round(3) #记录总比特数（四舍五入到小数点后三位）。
        results["bits"] = sum(bits).round(3)
        results["bpp"] = sum(bpps).round(3)#记录平均比特数（四舍五入到小数点后三位）。
        results["bpp(coords)"] = bpps[0] #记录坐标的平均比特数。
        results["bpp(feats)"] = bpps[1]#记录特征的平均比特数。
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec
        if idx == 0:
            all_results = results.copy(deep=True) #如果是第一个检查点，将结果复制给 all_results。
        else: 
            all_results = all_results.append(results, ignore_index=True)#否则，将结果追加到 all_results 中。
        csv_name = os.path.join(resultdir, os.path.split(filedir)[-1].split('.')[0]+'.csv')#构建结果文件的CSV文件名。
        all_results.to_csv(csv_name, index=False)#将所有结果写入CSV文件。
        print('Wrile results to: \t', csv_name)#打印结果写入的文件名。

    return all_results
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='../../../testdata/8iVFB/longdress_vox10_1300.ply')
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    ckptdir_list = ['./ckpts/r1_0.025bpp.pth', './ckpts/r2_0.05bpp.pth', 
                    './ckpts/r3_0.10bpp.pth', './ckpts/r4_0.15bpp.pth', 
                    './ckpts/r5_0.25bpp.pth', './ckpts/r6_0.3bpp.pth', 
                    './ckpts/r7_0.4bpp.pth']#指定了一系列检查点目录。

    all_results = test(args.filedir, ckptdir_list, args.outdir, args.resultdir, scaling_factor=args.scaling_factor, rho=args.rho, res=args.res)#调用 test 函数，对输入数据进行编码和解码，并记录结果。

    # plot RD-curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color='blue') #绘制RD曲线，包括D1和D2的PSNR值。
    filename = os.path.split(args.filedir)[-1][:-4]
    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.') #显示网格。
    plt.legend(loc='lower right')#显示图例。
    fig.savefig(os.path.join(args.resultdir, filename+'.jpg'))#将图保存为文件。

resolution（分辨率）: 这个参数通常用于指定生成的图像或数据的详细程度。在这个上下文中，它可能是指点云数据的分辨率。
bpp（Bits Per Pixel）: 这是一个衡量图像质量的指标，表示每个像素用多少比特来存储。在这个代码中，bpp是用来计算和绘制RD曲线（Rate-Distortion curve，比特率-失真曲线）的。
res（Resolution）: 这个参数和上面的resolution是一样的，都是指分辨率。
rho：这个参数在代码中的解释是"the ratio of the number of output points to the number of input points"，也就是输出点数和输入点数的比例。
p2point（Point-to-Point）和p2plane（Point-to-Plane）: 这两个参数在计算点云数据的误差时使用。p2point是指点对点的误差，p2plane是指点对平面的误差。在这个代码中，它们被用来计算和绘制RD曲线。
mse1 (p2point)：这是点对点（Point-to-Point）误差的均方误差（Mean Squared Error）。它是通过计算输入点云和输出点云之间每个点的距离的平方和的平均值得到的。
mse1,PSNR (p2point)：这是点对点误差的峰值信噪比（Peak Signal-to-Noise Ratio）。它是一个常用的评价图像质量的指标，计算方法是20乘以对数（以10为底）的信号最大可能功率和均方误差的比值。
mse1 (p2plane)：这是点对平面（Point-to-Plane）误差的均方误差。它是通过计算输入点云中的每个点到输出点云中最近平面的距离的平方和的平均值得到的。
mse1,PSNR (p2plane)：这是点对平面误差的峰值信噪比。
h. 1(p2point)：这个参数的具体含义可能需要更多的上下文信息，但从名字来看，它可能是某种基于点对点距离的度量。
h.,PSNR 1(p2point)：这个参数的具体含义也可能需要更多的上下文信息，但从名字来看，它可能是某种基于点对点距离的峰值信噪比度量。
