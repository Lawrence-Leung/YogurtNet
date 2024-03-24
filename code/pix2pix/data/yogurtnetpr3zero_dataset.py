import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np
import torchvision.transforms as transforms
import torch
from scipy.spatial import cKDTree
import random
from scipy.ndimage import median_filter

"""
    2023EDA挑战赛
    team magical_yogurt
    Lawrence Leung 2023/11/15
"""

class Yogurtnetpr3zeroDataset(BaseDataset):
    """
    YogurtNetcDataset 数据类
    从输入的PR，到输出的PR
    You can specify '--dataset_mode template' to use this model.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        增加新选项，改写旧选项
        我们的代码要求：
            1. input 3 channels（经过切片处理后）
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
        assert(opt.input_nc == 3
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

        A_index = [8, 9, 10, 11, 12, 13, 14, 15, 16]    # PR类通道，输入通道数为11
        #8,9,10,11 // 12,13,14 // 15,16
        A = self.YogurtextractIndex(A_index, A)     #萃取通道
        '''
        select_channel1 = [8, 9, 10, 11]
        weight1 = [1000, 1, 1, 1]
        A0 = self.weighted_sum_channels(A, select_channel1, weight1)

        select_channel1 = [12, 13, 14]
        weight1 = [1, 2, 2]
        A1 = self.weighted_sum_channels(A, select_channel1, weight1)

        select_channel1 = [15, 16]
        weight1 = [1, 1]
        A2 = self.weighted_sum_channels(A, select_channel1, weight1)

        A = np.dstack((A0, A1, A2)) #从第三个维度方向拼合起来
        
        A_index = [0, 1, 2]
        A_max = [(2e-08 * 10000 + 1.7e-04 + 1.4e-05 * 10 + 1.7e-04) / 8,
                 (720 + 430 * 2 + 390 * 2) / 2.5, (500 + 500) / 1.8]
        A_min = [0, 0, 0]
        '''
        select1test = [0, 1, 2, 3]
        weight1test = [1000, 1, 1, 1]
        A1test = self.weighted_sum_channels(A, select1test, weight1test)
        select2test = [5, 6, 7]
        weight2test = [1, 1, 1]
        A2test = self.weighted_sum_channels(A, select2test, weight2test)
        select3test = [7, 8]
        weight3test = [1, 1]
        A3test = self.weighted_sum_channels(A, select3test, weight3test)
        A = np.dstack((A1test, A2test, A3test))

        A_index = [0, 1, 2]
        A_max = [(2e-08 * 1000 + 1.7e-04 + 1.4e-05+ 1.7e-04),( 20+ 430 + 390), (500 + 500)]
        A_min = [0, 0, 0]

        '''
        A_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        A_max = [2e-08, 1.7e-04, 1.4e-05, 1.7e-04, 720, 430, 390, 500, 500]
        A_min = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        '''

        A = self.YogurtLinearNormalize(A, A_index, A_max, A_min)  #线性拉伸
        A = np.clip(A, -1, 1)   #大于1、小于-1的值，全部限制住


        B_index = [0, 1]    # 实际P坐标、实际R坐标，输出通道数为2
        B = self.YogurtextractIndex(B_index, B)      #萃取通道

        B_max = [0.108, 0.096]
        B_min = [0, 0]
        B = self.YogurtLinearNormalize(B, B_index, B_max, B_min)  # 97.5%分位点，最多仅损失2.5%

        B = np.clip(B, -1, 1)

        A = self.FIX_fill_matrix(A)     #填充数据
        B = self.FIX_fill_matrix(B)

        A, _ = self.filting_matrix(A)   #滤波处理
        B, _ = self.filting_matrix(B)   

        A = self.FIXprocess_matrix_edges1(A, 5)     #边缘处理
        B = self.FIXprocess_matrix_edges1(B, 5)
        
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

    def YogurtLinearNormalizeOld(self, matrix, modifyindex, max_val, min_val):
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
    '''
    def filting_matrix(self, origin_numpy, conv_size=7):
        # 获取原始矩阵的维度
        a, _, b = origin_numpy.shape

        # 创建一个空的numpy矩阵
        conv_numpy = np.zeros_like(origin_numpy)

        # 计算卷积
        for x in range(a):
            for y in range(a):
                for z in range(b):
                    # 计算滤波核的边界
                    x_min, x_max = max(0, x - conv_size // 2), min(a, x + conv_size // 2 + 1)
                    y_min, y_max = max(0, y - conv_size // 2), min(a, y + conv_size // 2 + 1)

                    # 提取滤波核区域并计算平均值
                    conv_region = origin_numpy[x_min:x_max, y_min:y_max, z]
                    conv_numpy[x, y, z] = np.mean(conv_region)
        # 计算损失矩阵
        loss_numpy = origin_numpy - conv_numpy

        return conv_numpy, loss_numpy
    '''

    def filting_matrix_average(self, origin_numpy, conv_size=7):
        """
        加速版本的矩阵滤波函数。

        :param origin_numpy: 输入的NumPy矩阵，形状为(a, a, b)
        :param conv_size: 滤波核的大小，默认为7
        :return: 滤波后的NumPy矩阵和原始矩阵的差值矩阵
        """
        a, _, b = origin_numpy.shape
        conv_numpy = np.zeros_like(origin_numpy)

        # 预先计算滤波核的面积
        kernel_area = conv_size ** 2

        # 使用累积和技术进行加速
        padded_matrix = np.pad(origin_numpy, ((conv_size // 2, conv_size // 2),
                                              (conv_size // 2, conv_size // 2),
                                              (0, 0)), mode='constant')

        cumsum_matrix = np.cumsum(np.cumsum(padded_matrix, axis=0), axis=1)

        for x in range(a):
            for y in range(a):
                for z in range(b):
                    # 考虑边界情况
                    x_min, x_max = x, min(x + conv_size, a + conv_size // 2)
                    y_min, y_max = y, min(y + conv_size, a + conv_size // 2)

                    # 计算卷积区域的累积和
                    total = cumsum_matrix[x_max, y_max, z]
                    if x_min > 0: total -= cumsum_matrix[x_min - 1, y_max, z]
                    if y_min > 0: total -= cumsum_matrix[x_max, y_min - 1, z]
                    if x_min > 0 and y_min > 0: total += cumsum_matrix[x_min - 1, y_min - 1, z]

                    # 计算平均值
                    conv_numpy[x, y, z] = total / kernel_area

        # 计算损失矩阵
        loss_numpy = origin_numpy - conv_numpy

        return conv_numpy, loss_numpy

    def filting_matrix_median_old(self, origin_numpy, conv_size=7):
        """
        使用中位数滤波的矩阵滤波函数。

        :param origin_numpy: 输入的NumPy矩阵，形状为(a, a, b)
        :param conv_size: 滤波核的大小，默认为7
        :return: 滤波后的NumPy矩阵和原始矩阵的差值矩阵
        """
        a, _, b = origin_numpy.shape
        conv_numpy = np.zeros_like(origin_numpy)

        padded_matrix = np.pad(origin_numpy, ((conv_size // 2, conv_size // 2),
                                            (conv_size // 2, conv_size // 2),
                                            (0, 0)), mode='constant')

        for x in range(a):
            for y in range(a):
                for z in range(b):
                    # 计算卷积核覆盖区域的边界
                    x_min, x_max = max(x - conv_size // 2, 0), min(x + conv_size // 2 + 1, a)
                    y_min, y_max = max(y - conv_size // 2, 0), min(y + conv_size // 2 + 1, a)

                    # 提取卷积核覆盖区域的所有像素值
                    kernel_values = padded_matrix[x_min:x_max, y_min:y_max, z]

                    # 计算中位数
                    conv_numpy[x, y, z] = np.median(kernel_values)

        # 计算损失矩阵
        loss_numpy = origin_numpy - conv_numpy

        return conv_numpy, loss_numpy

    def filting_matrix(self, origin_numpy, conv_size=7):
        """
        使用中位数滤波的矩阵滤波函数（优化版）。

        :param origin_numpy: 输入的NumPy矩阵，形状为(a, a, b)
        :param conv_size: 滤波核的大小，默认为7
        :return: 滤波后的NumPy矩阵和原始矩阵的差值矩阵
        """
        # 对每个通道使用median_filter进行中位数滤波
        conv_numpy = np.array([median_filter(origin_numpy[:, :, z], size=conv_size) for z in range(origin_numpy.shape[2])])
        conv_numpy = np.moveaxis(conv_numpy, 0, -1)

        # 计算损失矩阵
        loss_numpy = origin_numpy - conv_numpy

        return conv_numpy, loss_numpy

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

    def weighted_sum_channels(self, origin_numpy, select_channel, channel_multiply):
        """
        对NumPy矩阵的指定通道进行加权求和。

        :param origin_numpy: 输入的NumPy矩阵，形状为(a, a, b)
        :param select_channel: 选择的通道列表
        :param channel_multiply: 对应通道的加权系数列表
        :return: 加权求和后的NumPy矩阵，形状为(a, a, 1)
        """
        if len(select_channel) != len(channel_multiply):
            raise ValueError("select_channel and channel_multiply must be the same length.")

        # 初始化新矩阵
        a = origin_numpy.shape[0]
        new_numpy = np.zeros((a, a, 1))

        # 遍历选定通道并进行加权求和
        for i in range(len(select_channel)):
            channel_index = select_channel[i]
            weight = channel_multiply[i]
            new_numpy[:, :, 0] += origin_numpy[:, :, channel_index] * weight

        return new_numpy

    def fill_matrix_single(self, matrix):
        '''
            单数据点填充网格
        '''
        a, _, b = matrix.shape
        non_zero_points = np.argwhere(matrix.any(axis=2))  # 获取非零数据点的坐标
        zero_points = np.argwhere(~matrix.any(axis=2))    # 获取零数据点的坐标

        if non_zero_points.size == 0 or zero_points.size == 0:
            return matrix  # 如果没有非零点或零点，则不进行处理

        # 使用k-d树来找到空间上邻近的非零点
        tree = cKDTree(non_zero_points[:, :2])
        for point in zero_points:
            _, idx = tree.query(point[:2], k=1)  # 找到最近的非零点
            closest_non_zero_point = non_zero_points[idx]
            # 将非零点的数据复制到零点位置
            matrix[point[0], point[1], :] = matrix[closest_non_zero_point[0], closest_non_zero_point[1], :]

        return matrix

    def FIX_fill_matrix(self, matrix, c=10, seed=42):
        a, _, b = matrix.shape
        c = min(c, a)  # 确保c不超过矩阵边界

        # 生成非零数据块的中心点
        non_zero_blocks = [(i, j) for i in range(c//2, a - c//2) for j in range(c//2, a - c//2)
                        if np.any(matrix[i-c//2:i+c//2+1, j-c//2:j+c//2+1, :])]

        if not non_zero_blocks:
            return matrix  # 如果没有合适的非零块，则不进行处理

        random.seed(seed)  # 设置随机种子以保持结果的一致性

        for i in range(a):
            for j in range(a):
                if not np.any(matrix[i, j, :]):  # 如果是零数据点
                    chosen_block = random.choice(non_zero_blocks)  # 随机选择一个非零块
                    x, y = chosen_block
                    matrix[i-c//2:i+c//2+1, j-c//2:j+c//2+1, :] = matrix[x-c//2:x+c//2+1, y-c//2:y+c//2+1, :]

        return matrix

    def FIXprocess_matrix_edges(self, matrix, c=2):
        a, _, b = matrix.shape
        # 处理矩阵的顶部和底部边缘
        matrix[:c, :, :] = 0
        matrix[a-c:, :, :] = 0

        # 处理矩阵的左侧和右侧边缘
        matrix[:, :c, :] = 0
        matrix[:, a-c:, :] = 0

        return matrix

    def FIXprocess_matrix_edges1(self, matrix, c=2):
        a, _, b = matrix.shape
        # 处理矩阵的顶部和底部边缘
        matrix[:c, :, :] = -1       #注意：clip之后到-1，不是1
        matrix[a-c:, :, :] = -1

        # 处理矩阵的左侧和右侧边缘
        matrix[:, :c, :] = -1
        matrix[:, a-c:, :] = -1

        return matrix