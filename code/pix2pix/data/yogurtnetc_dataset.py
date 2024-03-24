import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np
import torchvision.transforms as transforms
import torch
"""
    2023EDA挑战赛
    team magical_yogurt
    Lawrence Leung 2023/11/15
"""

class YogurtnetcDataset(BaseDataset):
    """
    YogurtNetcDataset 数据类
    从输入的C，到输出的C
    You can specify '--dataset_mode template' to use this model.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        增加新选项，改写旧选项
        我们的代码要求：
            1. input 6 channels（经过切片处理后）
            2. output 2 channels（经过切片处理后）
            3. 模式：从A到B
            4. 使用所有的两块GPU进行训练
        set_defaults所有的参数在/torchnet/options/base_options.py代码文件中有查询。
        """
        # 覆盖并设置原有的参数，返回一个传参集合：parser
        parser.set_defaults(input_nc = 6, output_nc = 2, direction='AtoB', gpu_ids = '0,1')
        return parser

    def __init__(self, opt):
        """
        类构造函数。具有以下功能：保存选项，获取路径目录。
        上述三个功能都是可选的，可以在此构造函数中完成。
        我们知道，YogurtnetDataset是继承自BaseDataset的，
        BaseDataset有一个成员变量，名为opt。opt在base_dataset.py中有描述。
        opt是继承于BaseOptions类的子类。
        """
        # 首先运行基类的构造函数
        BaseDataset.__init__(self, opt)

        # ('--dataroot', required=True, help='path to images
        #       (should have subfolders trainA, trainB, valA, valB, etc)')
        # ('--phase', type=str, default='train', help='train, val, test, etc')
        """
            dataroot变量：第一个字段，由train.py传入
            phase变量：第二个字段，加入到dataroot变量之后，作为目录名称。
        """
        # 从dataset中读取dataset目录
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # 原始的.npy文件位置列表
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        # 用于存储结果的字典，键为目录，值为包含两个文件路径的元组
        paired_files = {}
        # 遍历文件列表，组织成所需的结构
        for file_path in self.AB_paths:
            # 分离目录和文件名
            directory, filename = file_path.rsplit('/', 1)
            # 检查文件名并按目录分类
            if filename in ['static_ir_POSTDATA.npy', 'static_ir_PREDATA.npy']:
                if directory not in paired_files:
                    paired_files[directory] = [None, None]  # 初始化元组占位
                # 根据文件名设置元组中的相应位置
                if filename == 'static_ir_POSTDATA.npy':
                    paired_files[directory][1] = file_path
                else:
                    paired_files[directory][0] = file_path
        # 从字典中提取最终的元组列表
        self.paired_paths = list(paired_files.values())

        # 确认参数信息
        assert(opt.input_nc == 6
               and opt.output_nc == 2
               and opt.direction == 'AtoB')

    def __getitem__(self, index):
        """
        返回每组数据点
        将每个A、B数据对，转化为PyTorch Tensor，最终放到Dict中去
        """
        path = self.paired_paths[index] #每组数据提取出来
        A = np.load(path[0])    #每组数据的A数据
        B = np.load(path[1])    #每组数据的B数据
        A_index = [0, 1, 2, 3, 4, 5]    # 聚类结果为2或3的C类通道，输入通道数为6
        B_index = [2, 3]    # 实际X坐标、实际Y坐标，输出通道数为2

        A = self.YogurtLinearNormalize(A, 0, 300, 0)    #拉伸，注意这个放在前面
        A = self.YogurtLinearNormalize(A, 1, 300, 0)
        A = self.YogurtLinearNormalize(A, 2, 300, 0)
        A = self.YogurtLinearNormalize(A, 3, 300, 0)
        A = self.YogurtLinearNormalize(A, 4, 300, 0)
        A = self.YogurtLinearNormalize(A, 5, 300, 0)
        B = self.YogurtLinearNormalize(B, 2, 300, 0)
        B = self.YogurtLinearNormalize(B, 3, 300, 0)

        A = self.YogurtextractIndex(A_index, A)         #提取
        B = self.YogurtextractIndex(B_index, B)

        """
        bug：
        这个错误信息表明您的网络输入（torch.cuda.DoubleTensor）和网络权重（torch.cuda.FloatTensor）的数据类型不匹配。
        在PyTorch中，通常网络权重是32位浮点数（即FloatTensor），而您的输入数据是64位浮点数（即DoubleTensor）。
        为了解决这个问题，您需要将输入数据的类型从DoubleTensor转换为FloatTensor。
        这可以通过使用.float()方法在将NumPy数组转换为PyTorch张量后立即进行。
        """
        A = torch.from_numpy(A).float() #转换为tensor
        B = torch.from_numpy(B).float()
        """
        bug：
        在PyTorch中，卷积神经网络通常期望输入的张量具有形状 (batch_size, channels, height, width)。
        在您的情况下，您的NumPy矩阵具有形状 (256, 256, 4)，
        这里的4很可能是通道数，而256x256是图像的高度和宽度。
        为了适应PyTorch的期望格式，您需要重新排列这些维度。
        使用permute方法将张量的维度从 (256, 256, 4) 变换为 (4, 256, 256)。
        """
        A = A.permute(2, 0, 1)
        B = B.permute(2, 0, 1)
        return {'A': A, 'B': B, 'A_paths': path[0], 'B_paths': path[1]}

    def __len__(self):
        """
        返回数据集的总数目
        """
        return len(self.paired_paths[0])

    def YogurtextractIndex(self, index_list, array):
        """
            从三维数组中按索引列表抽取切片并组合成新矩阵

            :param index_list: 包含索引的列表，索引对应于第三维度
            :param array: 三维的NumPy数组，形状为(256, 256, a)
            :return: 新的矩阵，形状为(256, 256, n)，其中n是index_list的长度
            """
        # 检查索引是否在数组的第三维度范围内
        if not all(0 <= idx < array.shape[2] for idx in index_list):
            raise ValueError("Index out of range in the third dimension.")

        # 使用列表推导式和np.stack组合所选的切片
        combined_array = np.stack([array[:, :, idx] for idx in index_list], axis=2)

        return combined_array

    def YogurtLinearNormalize(self, matrix, modifyindex, max_val, min_val):
        """
        对矩阵的指定通道进行线性标准化。

        :param matrix: 输入的NumPy矩阵，形状为(256, 256, a)
        :param modifyindex: 指定进行标准化的通道索引
        :param max_val: 该通道数据的最大值
        :param min_val: 该通道数据的最小值
        :return: 标准化后的NumPy矩阵
        """
        # 检查索引是否在第三维度的范围内
        if not 0 <= modifyindex < matrix.shape[2]:
            raise ValueError("Index out of range in the third dimension.")

        # 检查最大值和最小值是否有效
        if max_val <= min_val:
            raise ValueError("Max value must be greater than min value.")

        # 提取需要修改的通道
        channel = matrix[:, :, modifyindex]

        # 线性标准化
        normalized_channel = -1 + 2 * (channel - min_val) / (max_val - min_val)

        # 更新矩阵的对应通道
        output_matrix = matrix.copy()
        output_matrix[:, :, modifyindex] = normalized_channel

        return output_matrix

    def YogurtDeNormalize(self, matrix, modifyindex, max_val, min_val):
        """
        对标准化后的矩阵进行逆过程还原。

        :param matrix: 输入的NumPy矩阵，形状为(256, 256, a)
        :param modifyindex: 指定进行还原的通道索引
        :param max_val: 该通道数据的最大值
        :param min_val: 该通道数据的最小值
        :return: 还原后的NumPy矩阵
        """
        # 检查索引是否在第三维度的范围内
        if not 0 <= modifyindex < matrix.shape[2]:
            raise ValueError("Index out of range in the third dimension.")

        # 检查最大值和最小值是否有效
        if max_val <= min_val:
            raise ValueError("Max value must be greater than min value.")

        # 提取需要修改的通道
        normalized_channel = matrix[:, :, modifyindex]

        # 线性反标准化
        channel = ((normalized_channel + 1) / 2) * (max_val - min_val) + min_val

        # 更新矩阵的对应通道
        output_matrix = matrix.copy()
        output_matrix[:, :, modifyindex] = channel

        return output_matrix









