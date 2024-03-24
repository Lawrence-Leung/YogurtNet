'''
    core_pyramid_zero.py
    使用YogurtPyramid网络实现推理
    此文件应当被放在/yogurtpyramid文件夹中。
'''

import os
import numpy as np
import torch
import model.model as module_arch   #这个是调用本地子文件夹的数据。
import argparse
from parse_config import ConfigParser
import re
import sys

class pyramid:
    @classmethod
    def inputNumpyToTorchTensor(cls,input_numpy, B_max):
        '''
        输入Numpy矩阵，输出一个符合要求的PyTorch Tensor
        :param input_numpy: Numpy矩阵
        :return: 符合卷积网络输入要求的PyTorch Tensor
        '''
        A_index = [0, 1]
        # A_max = [0.179, 0.095]
        A_min = [0, 0]
        A = cls.YogurtLinearNormalize(input_numpy, A_index, B_max, A_min)
        A = torch.from_numpy(A).float()
        A = A.permute(2, 0, 1)
        A = A.unsqueeze(0) if A.dim() == 3 else A   #增加维度用unsqueeze
        return A
    @classmethod
    def outputTensorToNumpy(cls,output_tensor, B_max):
        '''
        输入PyTorch Tensor，输出一个符合要求的Numpy矩阵
        :param output_tensor: PyTorch Tensor
        :return: 符合输出要求的Numpy矩阵
        '''
        tensor = output_tensor.squeeze(0) if output_tensor.dim() == 4 else output_tensor    #减少维度用squeeze
        tensor = tensor.permute(1, 2, 0)  #重新排列维度以匹配(height, width, channels)
        tensor = tensor.numpy() #将PyTorch Tensor转换为Numpy Array
        B_index = [0, 1]
        # B_max = [0.179, 0.095]
        B_min = [0, 0]
        tensor = cls.YogurtLinearDenormalize(tensor, B_index, B_max, B_min) #别忘了这个！

        return tensor
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
    def realTimeSerialInference(cls,input_data, g_model, A_max, B_max):
        '''
        实时推理函数，串行推理
        :param input_data: 输入的Numpy矩阵
        :param g_model: 需要使用的生成器
        :return: 返回 输出的Numpy矩阵
        '''
        input_data = cls.inputNumpyToTorchTensor(input_data, B_max)  #先转换为能用的张量
        input_data = input_data.cuda()  #移动输入数据到GPU
        with torch.no_grad():   #推理，并且不计算梯度
            output_data = g_model(input_data)
        output_data = output_data.cpu() #将输出数据移动到CPU
        output_data = cls.outputTensorToNumpy(output_data, B_max)  #最后转换为numpy数组
        return output_data
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
    @classmethod
    def coreSerialInference(cls,entrance_path, IMG_SIZE, A_max, B_max, modelGparam, origin_input_tensor_dim_py):
        '''
        核心遍历+串行推理
        :param entrance_path: 推理文件入口地址
        :param modelGpath: GAN生成器模型地址
        :return: None
        '''

        '''
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available. Real-time inderence cannot proceed.")

        g_model = cls.loadModelG(modelGpath, config)
        g_model.eval()  #设置为评估模式（纯推理模式）
        g_model.cuda()  #移动模型到GPU
        '''

        predata_filename = 'static_ir_POSTDATApredmidPR.npy'
        outdata_filename = 'static_ir_POSTDATAfinal.npy'
        existindex = 0
        effectiveindex = 0
        predata_path = os.path.join(entrance_path, predata_filename)
        print(predata_path)

        #检查是否存在这个文件
        if os.path.exists(predata_path):
            input_data = np.load(predata_path)  #加载
            existindex += 1
            print(entrance_path, predata_filename)
            #检查形状是否匹配
            if input_data.shape == (IMG_SIZE, IMG_SIZE, origin_input_tensor_dim_py):
                print('yes')
                outputdata = cls.realTimeSerialInference(input_data, modelGparam, A_max, B_max)    #串行推理执行
                outputdata_path = os.path.join(entrance_path, outdata_filename)
                try:
                    np.save(outputdata_path, outputdata)    #逐一保存文件
                    effectiveindex += 1
                except Exception as e:
                    raise IOError('Inference Failed. ',e)
        print("All Finished! -------------------")
        print("Exist Indexes = ", existindex)
        print("Effective Indexes = ", effectiveindex)
        return
    @staticmethod
    def setconfig(g_modelfpu, 
                  g_modelnone, 
                  g_modelzero,
                  chip_type):
        # 参数路径，方法同上，取用脚本目录加相对路径方法
        if chip_type == 'zero-riscy':
            configpath = os.path.join(script_dir, 'config_zeroriscy.json')
            A_max = [(2e-08 * 1000 + 1.7e-04 + 1.4e-05 + 1.7e-04), (720 + 430 + 390), (500 + 500)]
            B_max = [0.108, 0.096]
            modelGpath = os.path.join(script_dir, 'allmodels/pyramid_zeroriscy/model_best_zero.pth')
            realGmodel = g_modelzero
        elif chip_type == 'none-riscy':
            configpath = os.path.join(script_dir, 'config_nonzero.json')
            A_max = [(2.40305e-08 * 1000 + 2.932e-04 + 1.4515e-05 * 10 + 2.932e-04), (2052.86 + 1791.71 + 1401.42),(1792.34 + 1441.00)]
            B_max = [0.179, 0.095]
            modelGpath = os.path.join(script_dir, 'allmodels/pyramid_nonzero/model_best_nonzero.pth')
            realGmodel = g_modelnone
        elif chip_type == 'riscy-fpu':
            configpath = os.path.join(script_dir, 'config_fpu.json')
            A_max = [(2.40305e-08 * 1000 + 3.0262e-04 + 1.4636e-05 * 10 + 3.0262e-04),(2049.44 + 1758.09 + 1323.34), (1800.80 + 1422.09)]
            B_max = [0.129, 0.099]
            modelGpath = os.path.join(script_dir, 'allmodels/pyramid_fpu/model_best_fpu.pth')
            realGmodel = g_modelfpu
        return configpath, A_max, B_max, modelGpath, realGmodel
    @classmethod
    def pyramid_main(cls,g_modelfpu,
                     g_modelnone,
                     g_modelzero,
                     entrance_path,
                     script_dir,
                     chip_type,
                     ):

        # 生成器路径,取用脚本目录加相对路径方法
        configpath, A_max, B_max, modelGpath, modelGparam = cls.setconfig(g_modelfpu, 
                                                                          g_modelnone, 
                                                                          g_modelzero,
                                                                          chip_type=chip_type)
        # 参数调用
        '''
        args = argparse.ArgumentParser(description='YogurtPyramid')

        args.add_argument('-c', '--config', default=configpath, type=str,help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=modelGpath, type=str,
                          help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default='0', type=str,
                          help='indices of GPUs to enable (default: all)')

        config = ConfigParser.from_args(args)
        '''
        #global output_tensor_dim_py,input_tensor_dim_py,origin_input_tensor_dim_py,IMG_SIZE
        output_tensor_dim_py = 2
        input_tensor_dim_py = 2
        origin_input_tensor_dim_py = 2
        IMG_SIZE = 256
        cls.coreSerialInference(entrance_path, IMG_SIZE, A_max, B_max, modelGparam, origin_input_tensor_dim_py)  #遍历+串行推理

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

def find_directories(root_path):
    subdirectories = []
    for root, dirs, files in os.walk(root_path):
        # If there are no subdirectories in the current directory
        if not dirs:
            subdirectories.append(root)
    return subdirectories

def loadModelG(modelGpath, config):
    '''
    加载生成器网络模型，并返回一个可以直接使用的网络模型
    :param modelGpath: 需要被使用的.pth
    :return: 一个generator_model，一个PyTorch模型
    '''
    """
    pyramid = model.YogurtPyramidModel256()
    try:    # 加载YogurtPyramid模型文件
        pyramid.load_state_dict(torch.load(modelGpath))
    except Exception as e:
        raise AttributeError("Unable to load model file", e)
    """

    pyramid = config.init_obj('arch', module_arch)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        pyramid = torch.nn.DataParallel(pyramid)
    pyramid.load_state_dict(state_dict)

    return pyramid

if __name__ == "__main__":
    # 获取script_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))

    args = argparse.ArgumentParser(description='YogurtPyramid')

    args.add_argument('-c', '--config', default=os.path.join(script_dir, 'config_zeroriscy.json'), type=str,help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=os.path.join(script_dir, 'allmodels/pyramid_zeroriscy/model_best_zero.pth'), type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                        help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
        # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. Real-time inderence cannot proceed.")

    g_modelzero = loadModelG(os.path.join(script_dir, 'allmodels/pyramid_zeroriscy/model_best_zero.pth'), config)
    g_modelzero.eval()  #设置为评估模式（纯推理模式）
    g_modelzero.cuda()  #移动模型到GPU

    g_modelnone = loadModelG(os.path.join(script_dir, 'allmodels/pyramid_nonzero/model_best_nonzero.pth'), config)
    g_modelnone.eval()  #设置为评估模式（纯推理模式）
    g_modelnone.cuda()  #移动模型到GPU

    g_modelfpu = loadModelG(os.path.join(script_dir, 'allmodels/pyramid_fpu/model_best_fpu.pth'), config)
    g_modelfpu.eval()  #设置为评估模式（纯推理模式）
    g_modelfpu.cuda()  #移动模型到GPUs

    # 输入具体路径（绝对路径，是遍历）
    entrance_path = '/home/stxianyx/code/eda/data/'  #entrance
    directories = []
    directories = find_directories(entrance_path)
    print(directories)
    for subdir in directories:
        print(subdir)
        # 参数
        chip_type = chiptype(subdir)
        pyramid.pyramid_main(g_modelfpu, 
                             g_modelnone, 
                             g_modelzero,
                             entrance_path=subdir,
                             script_dir=script_dir,
                             chip_type=chip_type, 
                             )

# 全局变量：output_tensor_dim_py,input_tensor_dim_py,origin_input_tensor_dim_py,IMG_SIZE
# pix2pix全局变量： output_tensor_dim,input_tensor_dim,origin_input_tensor_dim,IMG_SIZE,weight1test,weight2test,weight3test,A_max,B_max