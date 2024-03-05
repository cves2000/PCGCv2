import numpy as np 
import os, time
import pandas as pd
import subprocess
rootdir = os.path.split(__file__)[0]

#主要用于处理点云文件，并计算两个点云文件之间的误差

def get_points_number(filedir):
    plyfile = open(filedir)

    line = plyfile.readline()
    while line.find("element vertex") == -1:#find函数会返回子字符串在字符串中的起始位置，如果没有找到子字符串，就会返回-1
        line = plyfile.readline()#读取下一行
    number = int(line.split(' ')[-1][:-1])
    #这行代码的作用是从包含"element vertex"的行中提取出点的数量。具体来说：
    #- `line.split(' ')`：这一行使用空格将当前行分割成多个单词。
    #- `[-1]`：这一部分取出了最后一个单词，也就是点的数量。
    #- `[:-1]`：这一部分去掉了最后一个字符，因为在这种情况下，最后一个字符通常是换行符。
    #- `int(...)`：这一部分将点的数量转化为整数。
    # 所以，`number = int(line.split(' ')[-1][:-1])`这行代码的作用是提取出点的数量，并将其转化为整数。
    return number

def number_in_line(line):
#作用是从一个字符串中提取出数字，并返回该行最后一个数的浮点数类型    
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
    #在Python中，_, item是一种常见的用法，用于在遍历一个可迭代对象时忽略某些不需要的值。具体来说：
    # enumerate(wordlist)：这个函数会返回一个枚举对象，其中包含了wordlist中每个元素的索引和值。
    # _, item：这是一个元组，用于接收enumerate函数返回的索引和值。其中，_是一个常用的习惯用法，表示一个被忽略的值。在这个例子中，_接收了索引，但是我们并不需要使用这个索引，所以就用_来表示。而item则接收了wordlist中的每个元素。
    # 所以，_, item的意思就是在遍历wordlist时，忽略索引，只关注每个元素本身
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

def pc_error(infile1, infile2, res, normal=False, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
               "h.       1(p2point)", "h.,PSNR  1(p2point)" ]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)", 
               "h.       2(p2point)", "h.,PSNR  2(p2point)" ]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)", 
               "h.        (p2point)", "h.,PSNR   (p2point)" ]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF + haders_p2plane
    # headers1：这是一个包含了点对点（Point-to-Point）误差的均方误差（Mean Squared Error）和峰值信噪比（Peak Signal-to-Noise Ratio）的列表，以及Hausdorff距离和其对应的峰值信噪比。
    # headers2：这是一个类似于headers1的列表，但它是用来计算第二个点云数据的误差。
    # headersF：这是一个包含了点对点误差的均方误差和峰值信噪比，以及Hausdorff距离和其对应的峰值信噪比的列表。这些计算是基于输入的两个点云数据的。
    # haders_p2plane：这是一个包含了点对平面（Point-to-Plane）误差的均方误差和峰值信噪比的列表。
    # h：在这个上下文中，h可能是指Hausdorff距离。Hausdorff距离是一个用来衡量两个点集之间的距离的度量，它被定义为从一个点集到另一个点集的所有点的最短距离的最大值。
    # mse1和mse2对应的是两个点云数据集。具体来说：
    # mse1是指第一个点云数据集的误差。它是通过计算第一个点云数据集中的点与其对应的点（在第二个点云数据集中）之间的距离，然后取这些距离的平方的平均值得到的。
    # mse2是指第二个点云数据集的误差。它是通过计算第二个点云数据集中的点与其对应的点（在第一个点云数据集中）之间的距离，然后取这些距离的平方的平均值得到的。
    command = str(rootdir+'/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                        #   ' -n '+infile1+
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(res-1))
    # 这段代码主要是为了构造一个命令行命令，用于调用pc_error_d程序来计算两个点云文件之间的误差，并从程序的输出中提取出我们感兴趣的信息。具体来说：
    # headers1, headers2, headersF, haders_p2plane：这些变量定义了一些头部信息，这些信息将用于从pc_error_d程序的输出中提取结果。
    # headers = headers1 + headers2 + headersF + haders_p2plane：这一行将所有的头部信息合并到一个列表中。
    # command = str(rootdir+'/pc_error_d' + ' -a '+infile1+ ' -b '+infile2+ ' --hausdorff=1 '+ ' --resolution='+str(res-1))：这一行构造了一个命令行命令，该命令调用了一个名为pc_error_d的程序，并将两个点云文件的路径以及一些其他参数传递给这个程序。
    # 这段代码的目的是执行一个外部程序，并从其输出中提取出我们感兴趣的信息。
    if normal:
      headers += haders_p2plane
      command = str(command + ' -n ' + infile1)

    results = {}
   
    start = time.time()
    subp=subprocess.Popen(command, 
                          shell=True, stdout=subprocess.PIPE)

    c=subp.stdout.readline()
    # 这段代码的主要作用是执行一个外部程序（`pc_error_d`），并从其输出中提取结果。具体来说：
    # - `if normal: headers += haders_p2plane; command = str(command + ' -n ' + infile1)`：如果`normal`参数为True，那么就将`haders_p2plane`添加到`headers`中，并在命令中添加`-n`选项。
    # - `results = {}`：这一行初始化了一个空字典，用于存储结果。    
    # - `start = time.time()`：这一行记录了当前的时间，可以用于计算程序运行的时间。   
    # - `subp=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)`：这一行使用`subprocess.Popen`来执行命令，并将其输出重定向到一个管道。    
    # - `c=subp.stdout.readline()`：这一行读取了程序输出的第一行。   
    # 这段代码的目的是执行一个外部程序，并从其输出中提取出我们感兴趣的信息。
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c=subp.stdout.readline() 
    # print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return pd.DataFrame([results])#DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典（共同用一个索引）
    # 这段代码的主要作用是读取`pc_error_d`程序的输出，并从中提取出我们感兴趣的信息。具体来说：    
    # - `while c:`：这一行开始了一个循环，只要`c`（即程序的输出）不为空，就会继续执行循环。  
    # - `line = c.decode(encoding='utf-8')`：这一行将`c`解码为UTF-8格式的字符串。    
    # - `if show: print(line)`：如果`show`参数为True，那么就打印出当前的行。    
    # - `for _, key in enumerate(headers): if line.find(key) != -1: value = number_in_line(line); results[key] = value`：这一部分的代码遍历`headers`中的每个元素，如果当前的行中包含了这个元素，那么就从这一行中提取出数字，并将其存储在`results`字典中。
    # - `c=subp.stdout.readline()`：这一行在循环中读取了下一行。
    # - `return pd.DataFrame([results])`：这一行返回一个包含所有结果的`pandas.DataFrame`。
    # 这段代码的目的是执行一个外部程序，并从其输出中提取出我们感兴趣的信息。
