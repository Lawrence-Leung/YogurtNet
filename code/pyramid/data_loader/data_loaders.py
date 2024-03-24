import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import json

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class YogurtPyramidDataLoader(BaseDataLoader):
    """
    YogurtPyramid Dataloader
    Usage of DataLoader:
        BaseDataLoader us an iterator, to iterate through batches:
            for batch_idx, (x_batch, y_batch) in data_loader:
                pass
    """
    def __init__(self,
                 data_dir,      #数据入口目录
                 batch_size,    #batch大小
                 shuffle=True,  #是否随机
                 validation_split=0.0,  #验证集分配比例
                 num_workers=0, #workers数量
                 ):
        #我们不使用Transforms方法来实现数据处理，而是自己在__getitem__方法中实现。
        self.data_dir = data_dir
        self.dataset = YogurtPyramidDataset(self.data_dir) #注意，这里我们用类的方式组织！
        #这个类的getitem方法，所得到的是一组(A,B)的tuple对！
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        """
        需要注意：
        在 PyTorch 中，DataLoader 的 batch_size 参数和您在 split_and_stack 函数中
        生成的小矩阵（batch）的数量是两个不同的概念。
        DataLoader 的 batch_size 决定了每次模型训练迭代中使用多少个我自己定义的batch（小矩阵集合），
        而每个集合（小矩阵集合）的大小则由您的 split_and_stack 函数决定，这取决于原始大矩阵的大小。
        """

class YogurtPyramidDataset(Dataset):
    """
    一个数据集类，是继承自torch.utils.Dataset的类。
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Directory with all the data pairs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir    #源文件目录
        self.data_pairs = self.load_data_pairs(data_dir)    #加载AB数据对，用于训练 
        super().__init__()

    def load_data_pairs(self, data_dir):
        """
        根据自定义的数据存储格式加载数据对
        """
        results = []
        for root, dirs, files in os.walk(data_dir):

            # 检查当前目录下是否同时存在这两个文件
            
            if "static_ir_POSTDATApredmidPR.npy" in files and "static_ir_POSTDATA.npy" in files:
                # 加载文件为numpy数组
                A = np.load(os.path.join(root, "static_ir_POSTDATApredmidPR.npy"))  #A矩阵，用于训练输入
                B = np.load(os.path.join(root, "static_ir_POSTDATA.npy"))   #B矩阵，用于训练输出
                B = B[:,:,:2]
                # 检查两个文件的shape是否相同，且前两个维度是否为128的整数倍
                if A.shape == B.shape and A.shape[0] % 128 == 0 and A.shape[1] % 128 == 0:
                    results.append((A, B))
        
        #raise RuntimeError(f"Hello!{results} " )
        #补充
        return results  #返回一个带有[(A,B), (A,B), ...]的list

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        得到一组A、B对数据，其中idx是索引index。
        """
        A, B = self.data_pairs[idx]
        A_index = [0, 1]        #拉伸处理，拉伸从-1到1之间
        A_max = [0.179, 0.095]
        A_min = [0, 0]
        A = self.YogurtLinearNormalize(A, A_index, A_max, A_min)
        B = self.YogurtLinearNormalize(B, A_index, A_max, A_min)

        A = torch.from_numpy(A).float()     #转换为Float类型的Tensor
        B = torch.from_numpy(B).float()

        A = A.permute(2, 0, 1)
        B = B.permute(2, 0, 1)

        return A, B #注意，A和B不一样！

    def YogurtLinearNormalize(self, matrix, modifyindices, max_vals, min_vals):
        """
        对矩阵的指定多个通道进行线性标准化。

        :param matrix: 输入的NumPy矩阵，形状为(256, 256, a)
        :param modifyindices: 一个列表，包含要进行标准化的通道索引
        :param max_vals: 一个列表，包含每个通道数据的最大值
        :param min_vals: 一个列表，包含每个通道数据的最小值
        :return: 标准化后的NumPy矩阵
        """
        if len(modifyindices) != len(max_vals) or len(max_vals) != len(min_vals):
            raise ValueError("Length of lists modifyindices, max_vals, and min_vals must be equal.")
        # 创建输出矩阵的副本
        output_matrix = matrix.copy()

        for i in range(len(modifyindices)):
            modifyindex = modifyindices[i]
            max_val = max_vals[i]
            min_val = min_vals[i]

            # 检查索引是否在第三维度的范围内
            if not 0 <= modifyindex < matrix.shape[2]:
                raise ValueError("Index out of range in the third dimension for index " + str(modifyindex))

            # 检查最大值和最小值是否有效
            if max_val <= min_val:
                raise ValueError("Max value must be greater than min value for index " + str(modifyindex))

            # 提取需要修改的通道
            channel = matrix[:, :, modifyindex]

            # 线性标准化
            normalized_channel = -1 + 2 * (channel - min_val) / (max_val - min_val)

            # 更新矩阵的对应通道
            output_matrix[:, :, modifyindex] = normalized_channel

        return output_matrix

def split_and_stack_slow(matrix):
    """
    Split a matrix into 128x128x2 matrices and stack them into a batch.
    """
    a, _, _ = matrix.shape
    num_splits = a // 128
    split_matrices = []

    for i in range(num_splits):
        for j in range(num_splits):
            # Extract 128x128x2 matrix
            split_matrix = matrix[i*128:(i+1)*128, j*128:(j+1)*128, :]

            # Permute dimensions to make it compatible with neural network input
            split_matrix = split_matrix.permute(2, 0, 1)

            split_matrices.append(split_matrix)

    # Stack all split matrices to form a batch
    batch_matrix = torch.stack(split_matrices)

    return batch_matrix

def split_and_stack(matrix):
    """
    Optimized function to split a matrix into 128x128x2 matrices and stack them into a batch.
    """
    a, _, _ = matrix.shape
    num_splits = a // 128

    # Reshape and then transpose to achieve the necessary permutation and split
    batch_matrix = matrix.reshape(num_splits, 128, num_splits, 128, 2)
    """
    bug：
    在PyTorch中，卷积神经网络通常期望输入的张量具有形状 (batch_size, channels, height, width)。
    在您的情况下，您的NumPy矩阵具有形状 (256, 256, 4)，
    这里的4很可能是通道数，而256x256是图像的高度和宽度。
    为了适应PyTorch的期望格式，您需要重新排列这些维度。
    使用permute方法将张量的维度从 (256, 256, 4) 变换为 (4, 256, 256)。
    当然，下面的方法是原有方法经过加快之后的结果。
    """
    batch_matrix = batch_matrix.transpose(0, 2, 4, 1, 3)
    batch_matrix = batch_matrix.reshape(-1, 2, 128, 128)

    return batch_matrix
# batch_A = self.split_and_stack(A)   #分离，并堆叠成为一个128*128的batch
# batch_B = self.split_and_stack(B)