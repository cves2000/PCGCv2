#主要是用于3D点云的压缩和解压缩。它使用了MPEG G-PCCv12，这是一个用于无损压缩点云的工具
import os
import numpy as np 
import subprocess
rootdir = os.path.split(__file__)[0]
# 这行代码是用来获取当前Python脚本文件（__file__）的目录路径的。os.path.split(__file__)[0]会将__file__的路径分割成目录和文件名两部分，并返回目录部分。这样，rootdir就被设置为了当前脚本的目录路径。
# 这意味着，如果tmc3程序位于与你的Python脚本相同的目录下，那么这段代码就能正确地找到tmc3程序的路径。
def gpcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv12. 
    You can download and install TMC13 from 
    https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=0' + 
                            ' --positionQuantizationScale=1' + 
                            ' --trisoupNodeSizeLog2=0' + 
                            ' --neighbourAvailBoundaryLog2=8' + 
                            ' --intra_pred_max_node_size_log2=6' + 
                            ' --inferredDirectCodingMode=0' + 
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --uncompressedDataPath='+filedir + 
                            ' --compressedStreamPath='+bin_dir, 
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    
    return 
    # 这段代码是用来调用外部程序tmc3进行点云数据的压缩。
    # subprocess.Popen：这是Python的一个库函数，用于创建新的进程，并连接到该进程的输入/输出/错误管道，可以从中读取或写入数据。
    # rootdir+'/tmc3'：这是tmc3程序的路径，rootdir是程序的根目录。 
    # ' --mode=0'等：这些是tmc3程序的参数。例如，--mode=0表示运行模式为0（压缩模式），--positionQuantizationScale=1设置位置量化比例为1，等等。
    # ' --uncompressedDataPath='+filedir：这是输入的未压缩点云数据文件的路径。
    # ' --compressedStreamPath='+bin_dir：这是压缩后的数据文件的输出路径。
    # shell=True：这个参数允许你直接在系统shell中运行命令。
    # stdout=subprocess.PIPE：这个参数将新进程的标准输出重定向到一个管道，Python可以通过这个管道读取新进程的输出。
    # c=subp.stdout.readline()：这行代码读取tmc3程序的一行输出。
    # while c:：这个循环会一直读取tmc3程序的输出，直到没有更多的输出。
    # if show: print(c)：如果show参数为True，则将tmc3程序的输出打印到控制台。
def gpcc_decode(bin_dir, rec_dir, show=False):
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=1'+ 
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+rec_dir+
                            ' --outputBinaryPly=0'
                          ,
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)      
        c=subp.stdout.readline()
    
    return
    # 这段代码是用来调用外部程序tmc3进行点云数据的解压缩。让我们详细看一下：
    # subprocess.Popen：这是Python的一个库函数，用于创建新的进程，并连接到该进程的输入/输出/错误管道，可以从中读取或写入数据。
    # rootdir+'/tmc3'：这是tmc3程序的路径，rootdir是程序的根目录。
    # ' --mode=1'等：这些是tmc3程序的参数。例如，--mode=1表示运行模式为1（解压缩模式），--compressedStreamPath是输入的压缩后的二进制文件路径，--reconstructedDataPath是解压缩后的点云数据的输出路径，--outputBinaryPly=0表示输出的点云数据格式不是二进制的PLY格式。
    # shell=True：这个参数允许你直接在系统shell中运行命令。
    # stdout=subprocess.PIPE：这个参数将新进程的标准输出重定向到一个管道，Python可以通过这个管道读取新进程的输出。
    # c=subp.stdout.readline()：这行代码读取tmc3程序的一行输出。
    # while c:：这个循环会一直读取tmc3程序的输出，直到没有更多的输出。
    # if show: print(c)：如果show参数为True，则将tmc3程序的输出打印到控制台。
