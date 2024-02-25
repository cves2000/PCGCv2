#用于从3D网格模型中采样点云，并将这些点云保存为PLY或H5格式的文件
import open3d as o3d
import numpy as np
import random
import os, time
from data_utils import write_ply_ascii_geo, write_h5_geo

def sample_points(mesh_filedir, n_points=4e5, resolution=255):
    # 从网格模型中均匀采样点云
    # sample points uniformly.
    mesh = o3d.io.read_triangle_mesh(mesh_filedir)
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    except:
        print("ERROR sample_points", '!'*8)
        return
    points = np.asarray(pcd.points)
    return points

def get_rotate_matrix():
    # 随机生成旋转矩阵。
    m = np.eye(3,dtype='float32')
    m[0,0] *= np.random.randint(0,2)*2-1
    m = np.dot(m,np.linalg.qr(np.random.randn(3,3))[0])

    return m

def mesh2pc(mesh_filedir, n_points, resolution):
    # 生成点云。
    points = sample_points(mesh_filedir, n_points=n_points, resolution=resolution)
    # random rotate.随机旋转。
    points = np.dot(points, get_rotate_matrix())
    # normalize to fixed resolution. 归一化到固定分辨率。
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (resolution)
    # quantizate to integers.量化为整数
    points = np.round(points).astype('int')
    points = np.unique(points, axis=0)

    return points

def generate_dataset(mesh_filedirs, pc_rootdir, out_filetype, n_points=4e5, resolution=255):
    start_time = time.time()
    for idx, mesh_filedir in enumerate(mesh_filedirs):
        try: 
            points = mesh2pc(mesh_filedir, n_points, resolution)
        except:
            print("ERROR generate_dataset", idx, '!'*8)
            continue
        if out_filetype == 'ply':
            pc_filedir = os.path.join(pc_rootdir, str(idx) + '_' \
                                        + os.path.split(mesh_filedir)[-1].split('.')[0] + '.ply')
            write_ply_ascii_geo(pc_filedir, points)
        if out_filetype == 'h5':
            pc_filedir = os.path.join(pc_rootdir, str(idx) + '_' \
                                        + os.path.split(mesh_filedir)[-1].split('.')[0] + '.h5')
            write_h5_geo(pc_filedir, points)
        if idx % 100 == 0: print('='*20, idx, round((time.time() - start_time)/60.), 'mins', '='*20)

    return 
    # 这段代码是用于从3D网格模型中采样点云并将这些点云保存为PLY或H5格式文件的函数。让我详细解释一下：
    # 首先，我们传入了以下参数：
    # mesh_filedirs：包含网格模型文件路径的列表。
    # pc_rootdir：点云文件保存的根目录。
    # out_filetype：输出文件类型，可以是’ply’或’h5’。
    # n_points：采样的点数，默认为400,000。
    # resolution：归一化分辨率，默认为255。
    # 然后，我们遍历每个网格模型文件：
    # 通过调用 mesh2pc 函数，从网格模型中采样点云。
    # 如果 out_filetype 是 ‘ply’，则将点云保存为PLY格式文件。
    # 如果 out_filetype 是 ‘h5’，则将点云保存为H5格式文件。
    # 每处理100个模型，我们会打印一条进度信息。
    # 请注意，您需要将 mesh_filedirs 替换为您的网格模型文件路径，并根据需要调整其他参数。
def traverse_path_recursively(rootdir):
    filedirs = []
    def gci(filepath):
        files = os.listdir(filepath)
        for fi in files:
            fi_d = os.path.join(filepath,fi)            
            if os.path.isdir(fi_d):
                gci(fi_d)                  
            else:
                filedirs.append(os.path.join(filepath,fi_d))
        return
    gci(rootdir)

    return filedirs
    # traverse_path_recursively(rootdir) 函数用于递归遍历指定目录下的所有文件和子目录，并返回一个包含所有文件路径的列表。

if __name__ == "__main__":
    mesh_rootdir = "/home/ubuntu/HardDisk1/ModelNet40/"
    pc_rootdir = './dataset/'
    out_filetype = 'ply'
    out_filetype = 'h5'
    num_mesh = 100
    n_points = int(4e5) # dense
    resolution = 127

    input_filedirs = traverse_path_recursively(rootdir=mesh_rootdir)
    mesh_filedirs = [f for f in input_filedirs if (os.path.splitext(f)[1]=='.off' or os.path.splitext(f)[1]=='.obj')]# .off or .obj
    mesh_filedirs = random.sample(mesh_filedirs, num_mesh)
    print("mesh_filedirs:\n", len(input_filedirs), len(mesh_filedirs))
    if not os.path.exists(pc_rootdir): os.makedirs(pc_rootdir)

    generate_dataset(mesh_filedirs, pc_rootdir, out_filetype, n_points, resolution)
    # 在 if __name__ == "__main__": 语句块中，我们执行以下操作：
    # 设置网格模型文件的根目录 mesh_rootdir 和点云文件保存的根目录 pc_rootdir。
    # 指定输出文件类型为 ‘ply’ 或 ‘h5’。
    # 设置要处理的网格模型数量 num_mesh，采样点数 n_points，以及归一化分辨率 resolution。
    # 我们获取了所有网格模型文件的路径，并筛选出扩展名为 ‘.off’ 或 ‘.obj’ 的文件，存储在 mesh_filedirs 列表中。
    # 随机选择 num_mesh 个网格模型文件。
    # 如果点云文件保存的目录不存在，我们创建该目录。
    # 最后，我们调用 generate_dataset 函数，将每个网格模型转换为点云并保存为指定的文件类型。
    


