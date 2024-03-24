'''
    core_inferPR3_fpu.py
    核心串行推理代码
    此文件应当被放在/torchmodel文件夹中。
'''

import os
import numpy as np
import torch
import re
import argparse

from models.networks import define_G   #在torchmodel的models文件夹中，引入define_G函数
from collections import OrderedDict
import random
from scipy.ndimage import median_filter

class pix2pix_infer:

    @staticmethod
    def loadModelG(modelGpath):
        '''
        加载生成器网络模型，并返回一个可以直接使用的网络模型
        :param modelGpath: 需要被使用的.pth
        :return: 一个generator_model，一个PyTorch模型
        '''
        try:    # 加载生成器模型文件
            model_dict = torch.load(modelGpath)
        except Exception as e:
            raise AttributeError("Unable to load model file", e)
        new_dict = OrderedDict()
        for k, v in model_dict.items():
            # load_state_dict expects keys with prefix 'module'.
            new_dict["module." + k] = v
        #注意，define_G的参数需要和模型本身的参数高度一致!
        generator_model = define_G(input_nc = input_tensor_dim,       # 注意，我们的输入通道数为17
                                   output_nc = output_tensor_dim,       # 注意，我们的输出通道数为4
                                   ngf = 64,
                                   netG = "unet_256",   # 注意，pix2pix的核心架构是UNet_256，不是默认值！
                                   norm = "batch",
                                   use_dropout = True,
                                   init_gain = 0.02,
                                   gpu_ids = [0])
        generator_model.load_state_dict(new_dict)
        return generator_model
    @classmethod
    def inputNumpyToTorchTensor(cls,input_numpy):
        '''
        输入Numpy矩阵，输出一个符合要求的PyTorch Tensor
        :param input_numpy: Numpy矩阵
        :return: 符合卷积网络输入要求的PyTorch Tensor
        '''
        A_index = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # PR类通道，输入通道数为11
        A = cls.YogurtextractIndex(A_index, input_numpy)  # 萃取通道

        select1test = [0, 1, 2, 3]
        #weight1test = [1000, 1, 10, 1]
        A1test = cls.weighted_sum_channels(A, select1test, weight1test)
        select2test = [5, 6, 7]
        #weight2test = [1, 1, 1]
        A2test = cls.weighted_sum_channels(A, select2test, weight2test)
        select3test = [7, 8]
        #weight3test = [1, 1]
        A3test = cls.weighted_sum_channels(A, select3test, weight3test)
        A = np.dstack((A1test, A2test, A3test))

        A_index = [0, 1, 2]
        #A_max = [(2.40305e-08 * 1000 + 3.0262e-04 + 1.4636e-05 * 10+ 3.0262e-04),(2049.44 + 1758.09 + 1323.34), (1800.80 + 1422.09)]
        A_min = [0, 0, 0]


        A = cls.YogurtLinearNormalize(A, A_index, A_max, A_min)  # 线性拉伸
        A = np.clip(A, -1, 1)   #大于1、小于-1的值，全部限制住

        A = cls.FIX_fill_matrix(A)  #填充
        A, _ = cls.filting_matrix(A)  # 滤波处理
        A = cls.FIXprocess_matrix_edges(A, 3)  #边缘处理，使用3。


        A = torch.from_numpy(A).float()  #转换为Float，只有这样做才能让PyTorch读取
        A = A.permute(2, 0, 1)    #重新排列维度以匹配(channels, height, width)
        A = A.unsqueeze(0)    #修复以下报错：
        '''
        在PyTorch中，卷积神经网络的输入通常期待以下形式的四维张量：[batch_size, channels, height, width]。
        这里的batch_size可以是1，但是仍然需要这个维度存在。
        输入数据没有批次维度。即使你只有一个数据样本，你仍然需要通过unsqueeze(0)方法来增加一个批次维度。
        '''
        return A
    @classmethod
    def outputTensorToNumpy(cls,output_tensor):
        '''
        输入PyTorch Tensor，输出一个符合要求的Numpy矩阵
        :param output_tensor: PyTorch Tensor
        :return: 符合输出要求的Numpy矩阵
        '''
        # 如果存在批次维度，移除它
        if output_tensor.dim() == 4 and output_tensor.size(0) == 1: # 这个4不要轻易变动！
            output_tensor = output_tensor.squeeze(0)
        tensor = output_tensor.permute(1, 2, 0)  #重新排列维度以匹配(height, width, channels)
        tensor = tensor.numpy() #将PyTorch Tensor转换为Numpy Array
        tensor = cls.FIXprocess_matrix_edges1(tensor, 3)
        B_index = [0, 1]
        #B_max = [0.129, 0.099]
        B_min = [0, 0]
        tensor = cls.YogurtLinearDenormalize(tensor, B_index, B_max, B_min) #别忘了这个！

        #optmatrix = np.load('optMATRIX.npy')
        #tensor = custom_block_stretch(tensor, optmatrix)
        #tensor, _ = filting_matrix(tensor)
        return tensor
    @classmethod
    def realTimeSerialInference(cls,input_data, g_model):
        '''
        实时推理函数，串行推理
        :param input_data: 输入的Numpy矩阵
        :param g_model: 需要使用的生成器
        :return: 返回 输出的Numpy矩阵
        '''
        input_data = cls.inputNumpyToTorchTensor(input_data)  #先转换为能用的张量
        input_data = input_data.cuda()  #移动输入数据到GPU
        with torch.no_grad():   #推理，并且不计算梯度
            output_data = g_model(input_data)
        output_data = output_data.cpu() #将输出数据移动到CPU
        output_data = cls.outputTensorToNumpy(output_data)  #最后转换为numpy数组
        return output_data
    @staticmethod
    def YogurtextractIndex(index_list, array):
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
    @staticmethod
    def YogurtLinearNormalize(matrix, modifyindices, max_vals, min_vals):
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
    @staticmethod
    def weighted_sum_channels(origin_numpy, select_channel, channel_multiply):
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
    @staticmethod
    def FIX_fill_matrix(matrix, c=10, seed=42):
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
    @staticmethod
    def FIXprocess_matrix_edges(matrix, c=2):
        a, _, b = matrix.shape
        # 处理矩阵的顶部和底部边缘
        matrix[:c, :, :] = 0
        matrix[a-c:, :, :] = 0

        # 处理矩阵的左侧和右侧边缘
        matrix[:, :c, :] = 0
        matrix[:, a-c:, :] = 0

        return matrix
    @staticmethod
    def FIXprocess_matrix_edges1(matrix, c=2):
        a, _, b = matrix.shape
        # 处理矩阵的顶部和底部边缘
        matrix[:c, :, :] = -1
        matrix[a-c:, :, :] = -1

        # 处理矩阵的左侧和右侧边缘
        matrix[:, :c, :] = -1
        matrix[:, a-c:, :] = -1

        return matrix
    @staticmethod
    def filting_matrix(origin_numpy, conv_size=7):
        """
        使用中位数滤波的矩阵滤波函数（优化版）。

        :param origin_numpy: 输入的NumPy矩阵，形状为(a, a, b)
        :param conv_size: 滤波核的大小，默认为7
        :return: 滤波后的NumPy矩阵和原始矩阵的差值矩阵
        """
        # 对每个通道使用median_filter进行中位数滤波
        conv_numpy = np.array(
            [median_filter(origin_numpy[:, :, z], size=conv_size) for z in range(origin_numpy.shape[2])])
        conv_numpy = np.moveaxis(conv_numpy, 0, -1)

        # 计算损失矩阵
        loss_numpy = origin_numpy - conv_numpy

        return conv_numpy, loss_numpy
    @staticmethod
    def YogurtLinearDenormalize(matrix, modifyindices, max_vals, min_vals):
        """
        对矩阵的指定多个通道进行线性去标准化。

        :param matrix: 输入的NumPy矩阵，形状为(256, 256, a)
        :param modifyindices: 一个列表，包含要进行去标准化的通道索引
        :param max_vals: 一个列表，包含每个通道数据的最大值
        :param min_vals: 一个列表，包含每个通道数据的最小值
        :return: 去标准化后的NumPy矩阵
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
            normalized_channel = matrix[:, :, modifyindex]

            # 线性去标准化
            channel = min_val + ((normalized_channel + 1) * (max_val - min_val)) / 2

            # 更新矩阵的对应通道
            output_matrix[:, :, modifyindex] = channel

        return output_matrix
    @classmethod
    def coreSerialInference(cls,entrance_path, modelGpath):
        '''
        核心遍历+串行推理
        :param entrance_path: 推理文件入口地址
        :param modelGpath: GAN生成器模型地址
        :return: None
        '''
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available. Real-time inderence cannot proceed.")

        g_model = pix2pix_infer.loadModelG(modelGpath)
        g_model.eval()  #设置为评估模式（纯推理模式）
        g_model.cuda()  #移动模型到GPU

        predata_filename = 'static_ir_PREDATA.npy'
        outdata_filename = 'static_ir_POSTDATApredmidPR.npy'
        existindex = 0
        effectiveindex = 0
        for root, dirs, files in os.walk(entrance_path):
            predata_path = os.path.join(root, predata_filename)

            #检查是否存在这个文件
            if predata_filename in files:
                input_data = np.load(predata_path)  #加载
                existindex += 1
                print(root, outdata_filename)
                #检查形状是否匹配
                if input_data.shape == (IMG_SIZE, IMG_SIZE, origin_input_tensor_dim):
                    print('yes')
                    outputdata = cls.realTimeSerialInference(input_data, g_model)    #串行推理执行
                    outputdata_path = os.path.join(root, outdata_filename)
                    try:
                        np.save(outputdata_path, outputdata)    #逐一保存文件
                        effectiveindex += 1
                    except Exception as e:
                        raise IOError('Inference Failed. ',e)
        print("All Finished! -------------------")
        print("Exist Indexes = ", existindex)
        print("Effective Indexes = ", effectiveindex)
        return
    @classmethod
    def pix2pix_infer_main(cls,entrance_path,script_dir,chip_type):
        global output_tensor_dim,input_tensor_dim,origin_input_tensor_dim,IMG_SIZE,weight1test,weight2test,weight3test,A_max,B_max
        output_tensor_dim = 2
        input_tensor_dim = 3
        origin_input_tensor_dim = 17
        IMG_SIZE = 256
        if chip_type == 'zero-riscy':
            weight1test = [1000, 1, 1, 1]
            weight2test = [1, 1, 1]
            weight3test = [1, 1]
            A_max = [(2e-08 * 1000 + 1.7e-04 + 1.4e-05 + 1.7e-04), (720 + 430 + 390), (500 + 500)]
            B_max = [0.108, 0.096]
            modelGpath = os.path.join(script_dir, 'allmodels/p2p_zeroriscy/latest_net_G.pth')
        elif chip_type == 'none-riscy':
            weight1test = [1000, 1, 10, 1]
            weight2test = [1, 1, 1]
            weight3test = [1, 1]
            A_max = [(2.40305e-08 * 1000 + 2.932e-04 + 1.4515e-05 * 10 + 2.932e-04), (2052.86 + 1791.71 + 1401.42),
                     (1792.34 + 1441.00)]
            B_max = [0.179, 0.095]
            modelGpath = os.path.join(script_dir, 'allmodels/p2p_nonzero/latest_net_G.pth')
        elif chip_type == 'riscy-fpu':
            weight1test = [1000, 1, 10, 1]
            weight2test = [1, 1, 1]
            weight3test = [1, 1]
            A_max = [(2.40305e-08 * 1000 + 3.0262e-04 + 1.4636e-05 * 10 + 3.0262e-04),
                     (2049.44 + 1758.09 + 1323.34), (1800.80 + 1422.09)]
            B_max = [0.129, 0.099]
            modelGpath = os.path.join(script_dir, 'allmodels/p2p_fpu/latest_net_G.pth')        # 生成器路径
            
        cls.coreSerialInference(entrance_path, modelGpath)  # 遍历+串行推理

def chiptype(fpath):
    folder_name = fpath.split(sep='/')[-1]
    if re.compile(r'zero-riscy').search(folder_name):
        print('chip type is zero-riscy')
        return 'zero-riscy'
    elif re.compile(r'RISCY_freq').search(folder_name):
        print('chip type is none-zero')
        return 'none-zero'
    elif re.compile(r'RISCY-FPU').search(folder_name):
        print('chip type is riscy-fpu')
        return 'riscy-fpu'
    else:
        exit()

def get_path():
    parser = argparse.ArgumentParser(description='Add path Parser')
    parser.add_argument('--path', default='./data', type=str, help='root path of input files')
    args = parser.parse_args()
    print(args.path)
    return args.path


if __name__ == "__main__":
    # 脚本所在位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 遍历入口路径
    entrance_path = get_path() #'/home/stxianyx/code/eda/data/RISCY_FPU'
    # 芯片类型
    chip_type = chiptype(entrance_path)

    pix2pix_infer.pix2pix_infer_main(entrance_path=entrance_path,script_dir=script_dir,chip_type=chip_type)
