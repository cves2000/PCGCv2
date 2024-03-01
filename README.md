# Multiscale Point Cloud Geometry Compression

​	We apply an **end-to-end learning framework** to compress the 3D  point cloud geometry (PCG) efficiently. Leveraging the sparsity nature of point cloud, we introduce the **multiscale structure** to represent native PCG compactly, offering the **hierarchical reconstruction** capability via progressive learnt re-sampling. Under this framework, we devise the **sparse convolution-based autoencoder** for feature analysis and aggregation. At the bottleneck layer, geometric occupancy information is losslessly encoded with a very small percentage of bits consumption, and corresponding feature attributes are lossy compressed. 

这是一个关于多尺度点云几何压缩的项目。该项目应用了一个端到端的学习框架，有效地压缩3D点云几何（PCG）。利用点云的稀疏性质，我们引入了多尺度结构，紧凑地表示原生PCG，通过逐步学习的重采样提供分层重建能力。在这个框架下，我们设计了基于稀疏卷积的自动编码器进行特征分析和聚合。在瓶颈层，几何占用信息被无损编码，占用了非常小的比特消耗，相应的特征属性被有损压缩。
## News

- 2021.11.23 We proposed a better and unified PCGC framework based on PCGCv2, named **SparsePCGC**. It can support both **lossless** and lossy compression, as well as dense point clouds (e.g., 8iVFB) and sparse **LiDAR** point clouds (e.g., Ford). Here is the links: [paper](https://arxiv.org/abs/2111.10633) [code](https://github.com/NJUVISION/SparsePCGC)
- 2021.7.28 We have simplified the code, and use torchac to replace tensorflow-compression for arithmetic coding in the updated version.
- 2021.2.25 We have updated MinkowskiEngine to v0.5. The bug on GPU is fixed. And the encoding and decoding runtime is reduced.
- 2021.1.1 Our paper has been accepted by **DCC2021**! [[paper](https://arxiv.org/abs/2011.03799)]  [[presentation](https://sigport.org/documents/multiscale-point-cloud-geometry-compression)]

该项目的一些重要更新包括：
2021年11月23日，他们提出了一个基于PCGCv2的更好、更统一的PCGC框架，名为SparsePCGC。它可以支持无损和有损压缩，以及密集点云（例如，8iVFB）和稀疏LiDAR点云（例如，Ford）。
2021年7月28日，他们简化了代码，并使用torchac替换了tensorflow-compression进行算术编码。
2021年2月25日，他们更新了MinkowskiEngine到v0.5。GPU上的错误已经修复，编码和解码的运行时间也有所减少。
2021年1月1日，他们的论文被DCC2021接受！



## Requirments
- python3.7 or 3.8
- cuda10.2 or 11.0
- pytorch1.7 or 1.8
- MinkowskiEngine 0.5 or higher (for sparse convolution)
- torchac 0.9.3 (for arithmetic coding) https://github.com/fab-jul/torchac
- tmc3 v12 (for lossless compression of downsampled point cloud coordinates) https://github.com/MPEGGroup/mpeg-pcc-tmc13

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 

- Pretrained Models: https://box.nju.edu.cn/f/b2bc67b7baab404ea68b/ (or Baidu Netdisk link：https://pan.baidu.com/s/14kPV1ZJPWCsuGM0V6mWytg?pwd=pcgc)
- Testdata: https://box.nju.edu.cn/f/30b96fc6332646c5980a/ (or Baidu Netdisk link: https://pan.baidu.com/s/1ccLR1fBrupIOeb2_0fPvQw?pwd=pcgc)
- Training Dataset: https://box.nju.edu.cn/f/1ff01798d4fc47908ac8/

该项目的运行环境要求包括python3.7或3.8，cuda10.2或11.0，pytorch1.7或1.8，MinkowskiEngine 0.5或更高版本（用于稀疏卷积），torchac 0.9.3（用于算术编码），tmc3 v12（用于无损压缩下采样点云坐标）等。他们推荐你按照NVIDIA/MinkowskiEngine的指南来设置稀疏卷积的环境。
预训练模型、测试数据和训练数据集的链接已经在上面提供。
## Usage

### Testing
Please download the pretrained models and install tmc3 mentioned above first.
```shell
sudo chmod 777 tmc3 pc_error_d
python coder.py --filedir='longdress_vox10_1300.ply' --ckptdir='ckpts/r3_0.10bpp.pth' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='longdress_vox10_1300.ply' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='dancer_vox11_00000001.ply'--scaling_factor=1.0 --rho=1.0 --res=2048
python test.py --filedir='Staue_Klimt_vox12.ply' --scaling_factor=0.375 --rho=4.0 --res=4096
python test.py --filedir='House_without_roof_00057_vox12.ply' --scaling_factor=0.375 --rho=1.0 --res=4096
```
The testing rusults of 8iVFB can be found in `./results`

测试
首先，请下载预训练模型并安装上面提到的tmc3。
运行以下命令来测试模型：
sudo chmod 777 tmc3 pc_error_d
python coder.py --filedir='longdress_vox10_1300.ply' --ckptdir='ckpts/r3_0.10bpp.pth' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='longdress_vox10_1300.ply' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='dancer_vox11_00000001.ply'--scaling_factor=1.0 --rho=1.0 --res=2048
python test.py --filedir='Staue_Klimt_vox12.ply' --scaling_factor=0.375 --rho=4.0 --res=4096
python test.py --filedir='House_without_roof_00057_vox12.ply' --scaling_factor=0.375 --rho=1.0 --res=4096
你可以在./results目录下找到8iVFB的测试结果。
### Training
```shell
 python train.py --dataset='training_dataset_rootdir'
```
训练 运行以下命令来训练模型：
python train.py --dataset='training_dataset_rootdir'
请将training_dataset_rootdir替换为你的训练数据集的根目录。

## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help from Prof. Dandan Ding from Hangzhou Normal University and Prof. Zhu Li from University of Missouri at Kansas. Please contact us (mazhan@nju.edu.cn and wangjq@smail.nju.edu.cn) if you have any questions.
