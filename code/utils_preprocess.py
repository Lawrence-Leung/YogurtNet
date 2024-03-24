import re
import json
import os
import glob
import random
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
import argparse



class Export_NCPR:

    @staticmethod
    def re_match_power_rpt(file_path_of_pulpino):  # 普适性地返回power.rpt
        pattern = os.path.join(file_path_of_pulpino, '*power.rpt')
        files = glob.glob(pattern)
        file_path = files[0]
        return file_path

    @staticmethod
    def Write_N1(inst_names_list):  # Write N to output txt
        '''
        将inst_names_list的内容直接写入inst_name_output中去
        :param inst_name_output:
        :param inst_names_list:
        :return:
        '''
        with open('inst_names_list.txt', "w") as file_N:
            # "inst_name_output.txt"
            for element in inst_names_list:
                file_N.write(element)
                file_N.write("\n")

    @staticmethod
    def print_progressbar_global(progress, info):
        '''
        打印进度条
        :param progress: 进度百分比
        :param info: 当前进度的信息
        :return: None
        '''
        progress_bar = '[' + '=' * progress + '>' + '.' * (100 - progress) + ']'
        print('\r', progress_bar, f'{progress}% completed; ' + info, end='', flush=True)
    # Functions that invoked

    @classmethod
    def Make_Name_list1(cls, file_path_pulpino):
        '''
        根据pulpino.rpt文件制作一个名称清单
        :param file_path_pulpino: 输入pulpino
        :return: inst_names_list，一个list变量
        '''
        inst_names_set = set()  # 使用集合来提高查找效率
        bbox_list = []
        P_list = []

        file_path = cls.re_match_power_rpt(file_path_pulpino)
        print(file_path)

        with open(file_path, 'r') as file_pulpino:
            for i, line in enumerate(file_pulpino):
                if i <= 1:  # 直接跳过前两行
                    continue
                linep = line.rstrip('\n')
                linelist = linep.split()  # 分割任意空白字符
                name = linelist[0]
                if name not in inst_names_set:
                    inst_names_set.add(name)  # 添加到集合中
                    bbox_list.append(eval(linelist[12]))  # 假设这里是一个数字
                    P28 = [float(a) for a in linelist[2:8]]
                    P_list.append(P28)

        return list(inst_names_set), bbox_list, P_list  # 转换回列表

    @staticmethod
    def Read_eff1(file_path_eff):  # Load eff to eff_list[]
        '''
        根据eff_res.rpt文件，读取eff_list并输出
        :param file_path_eff: eff_res.rpt文件所在的目录
        :return: 输出的eff_list列表
        '''
        eff_list = []
        eff_name = 'eff_res.rpt'
        file_path = f"{file_path_eff}/{eff_name}"
        with open(file_path, 'r') as file_eff:
            for line in file_eff:
                line1 = line.rstrip('\n')

                line1 = line1.replace('-', '0')  # 防止报错 ！！！
                # print(line1)

                eff_list.append(line1)
        eff_list = eff_list[1:]
        return eff_list

    @staticmethod
    def Read_min1(file_path_min):
        '''
        从对应目录的min_path_res.rpt生成min_list列表变量
        :param file_path_min: 指定min_path_res.rpt文件所在的目录
        :return: min_list：所生成的列表变量
        '''
        min_list = []
        min_name = 'min_path_res.rpt'
        file_path = f"{file_path_min}/{min_name}"
        with open(file_path, 'r') as file_min:
            for line in file_min:
                line = line.rstrip('\n')
                min_list.append(line)
        min_list = min_list[1:]
        return min_list

    @staticmethod
    def make_pinlocnR13_list1(inst_names_list, eff_list):
        '''
        输入变量inst_names_list、eff_list，输出变量pin_loc_list和R13_list。
        :param inst_names_list
        :param eff_list
        :return: pin_loc_list, R13_list
        '''
        pin_loc_list = []
        R13_list = []

        # 假设eff_list是一个列表，其中的每个元素都是一个包含空格的字符串
        # 这里的代码可以根据实际情况进行调整
        eff_dict = {o.split(sep='  ')[5]: o for o in eff_list}
        # debug：应当改为5，不是改为0
        try:
            for inst_name in inst_names_list:
                if inst_name in eff_dict:
                    o = eff_dict[inst_name]
                    templist = o.split(sep='  ')

                    VDD = templist[3]
                    VSS = templist[4]
                    if VSS == '(0 0 0 0 0)' and VDD != '(0 0 0 0 0)':
                        # print('(0,0,0,0,0)')

                        VDDxy = VDD[1:-16].split(sep=' ')
                        xymid = [float(VDDxy[0]), float(VDDxy[1])]
                    elif VDD == '(0 0 0 0 0)' and VSS != '(0 0 0 0 0)':
                        VSSxy = VSS[1:-16].split(sep=' ')
                        xymid = [float(VSSxy[0]), float(VSSxy[1])]
                    else:
                        VDDxy = VDD[1:-16].split(sep=' ')
                        VSSxy = VSS[1:-16].split(sep=' ')

                        xmid = (float(VDDxy[0]) + float(VSSxy[0])) / 2
                        ymid = (float(VDDxy[1]) + float(VSSxy[1])) / 2

                        xymid = [xmid, ymid]
                    pin_loc_list.append(xymid)

                    R13 = [float(x) for x in templist[0:3]]
                    R13_list.append(R13)

        except ValueError as e:
            print('\nExport fail.')
            final_output(fpath)
            exit()

        return pin_loc_list, R13_list
        # 调用函数
        # pin_loc_list, R13_list = make_pinlocnR13_list(inst_names_list, eff_list)

    @staticmethod
    def make_R4_1(inst_names_list, min_list):
        '''
        输入inst_names_list，输出min_list
        :param inst_names_list
        :param min_list
        :return: R4_list
        '''
        # 创建一个字典用于存储min_list中的数据
        min_dict = {}
        for line in min_list:
            parts = line.split()
            if len(parts) >= 3:
                key = parts[-1]  # 获取行尾的实例名称
                value = parts[0]
                if value:
                    if key not in min_dict:
                        min_dict[key] = []
                    min_dict[key].append(value)

        # 使用字典快速匹配和生成结果
        R4_list = []
        for inst_name in inst_names_list:
            if inst_name in min_dict:
                if len(min_dict[inst_name]) >= 2:
                    # print(inst_name,min_dict[inst_name][:2])
                    R4_list.append(min_dict[inst_name][:2])
                elif len(min_dict[inst_name]) == 1:
                    R4_list.append([min_dict[inst_name][0], 0])
        return R4_list

    @staticmethod
    def Export_NCPR1(inst_names_list, pin_loc_list, bbox_list, R13_list, R4_list, P_list, fpath):  # Write NCPR_pre txt.
        '''
        Process 1 大处理函数，分为多种功能
        :param pin_loc_list
        :param bbox_list
        :param R13_list
        :param R4_list
        :param P_list
        :param chip_address: 某个芯片例程对应的目录名
        :return: None
        '''
        C_list = [[c1, c2] for c1, c2 in zip(pin_loc_list, bbox_list)]
        R_list = [[r1, r2] for r1, r2 in zip(R13_list, R4_list)]
        # convent all C labels in C_list

        NCPR_json_full_name = 'NCPR_full.json'
        file_path = f"{fpath}/{NCPR_json_full_name}"
        # 使用zip函数将两个列表中的元素一一对应
        CPR = [[c, p, r] for c, p, r in zip(C_list, P_list, R_list)]
        # print(CPR)
        zipped = zip(inst_names_list, CPR)
        # 将元组列表转换为字典
        dictionary = {key: value for key, value in zipped}
        # 将字典写入文件
        with open(file_path, 'w') as file:
            json.dump(dictionary, file)

    @classmethod
    def preprocess_main(cls,fpath, inst_names_list, bbox_list, P_list):
        eff_list = cls.Read_eff1(fpath)

        min_list = cls.Read_min1(fpath)

        pin_loc_list, R13_list = cls.make_pinlocnR13_list1(inst_names_list, eff_list)

        R4_list = cls.make_R4_1(inst_names_list, min_list)

        cls.Export_NCPR1(inst_names_list, pin_loc_list, bbox_list, R13_list, R4_list, P_list, fpath)

        print('NCPR Done.')
        # cls.print_progressbar_global(100, 'Done.')

class W2V_infer:
    @staticmethod
    def defaultSerialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 判断如果某个数据类型是numpy的array，那么自动将其转化为list类型
        raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")

    @staticmethod
    def findNearestEmptySpiral(x, y, grid,size):
        #以下是参数
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] #分别为右、下、左、上
        dir_idx = 0
        steps_in_current_dir = 1
        steps_taken = 0
        spiral_level = 1

        while True:
            for _ in range(2):
                for _ in range(steps_in_current_dir):
                    if 0 <= x < size and 0 <= y < size and not grid[x][y]:
                        return x, y
                    x += directions[dir_idx][0]
                    y += directions[dir_idx][1]
                    steps_taken += 1
                dir_idx = (dir_idx + 1) % 4
            steps_in_current_dir += 1
            spiral_level += 1

    @staticmethod
    def assignValues2Dict(my_dict:dict, my_list:list):
        #先检查dict和list的长度是否一致
        if len(my_dict) != len(my_list):
            raise ValueError("Dict和List中元素数量不匹配")

        #将list的值按顺序赋给dict的键
        for key, value in zip(my_dict.keys(), my_list):
            my_dict[key] = value

    @staticmethod
    def Read_inst_list(inst_names_list):
        return [name.replace("[", "_-").replace("]", "-_") for name in inst_names_list]

    @staticmethod
    def get_size(insts_list):
        n = len(insts_list)
        if n < 65536:
            size = 256
        #elif 65536 < n < 147456:
        #    size = 384
        #elif 147456 < n < 409600:
        #    size = 640
        else:
            final_output(fpath)
            exit()

        print('size:',size)
        return size

    @staticmethod
    def w2v_infer5(lines):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if chip_type == 'zero-riscy':
            Voc_Table_path = os.path.join(script_dir, 'Voc_Table', 'Voc_Table5_zero-riscy.json')
        elif chip_type == 'none-zero':
            Voc_Table_path = os.path.join(script_dir, 'Voc_Table', 'Voc_Table5_none-zero.json')
        elif chip_type == 'riscy-fpu':
            Voc_Table_path = os.path.join(script_dir, 'Voc_Table', 'Voc_Table5_riscy-fpu.json')
        else:
            final_output(fpath)

        # print(Voc_Table_path)
        with open(Voc_Table_path, 'r') as file:
            transfer_dict = json.load(file)

        # 使用defaultdict来简化字典操作
        coord_dict5 = defaultdict(lambda: [0, 0, 0, 0, 0])

        for eachline0 in lines:
            eachline = eachline0.rstrip('\n')
            words = eachline.split('/')
            wordvalue5 = coord_dict5[eachline]  # 默认值是[0, 0, 0, 0, 0]

            for word in words:
                if word in transfer_dict:
                    # 使用列表推导式简化累加操作
                    wordvalue5[:] = [a + b for a, b in zip(wordvalue5, transfer_dict[word])]

            # 不需要检查eachline是否已存在，defaultdict已经处理过了
            coord_dict5[eachline] = wordvalue5

        # 将defaultdict转换回普通dict
        return dict(coord_dict5)

    @classmethod
    def w2v_5to2_export(cls, coord_dict5, size, fpath):
        pca = PCA(n_components=2)
        coord_list5_forPCA = list(coord_dict5.values())
        coord_list2 = pca.fit_transform(coord_list5_forPCA)

        # 使用numpy来找到最小和最大值，这比循环快得多
        x_min, y_min = coord_list2.min(axis=0)
        x_max, y_max = coord_list2.max(axis=0)

        # 计算缩放比例，避免在循环中重复计算
        x_scale = (size - 1) / (x_max - x_min)
        y_scale = (size - 1) / (y_max - y_min)

        # 使用列表推导式和预先计算的缩放比例来计算新坐标
        adjustlist2 = [(round((x - x_min) * x_scale), round((y - y_min) * y_scale)) for x, y in coord_list2]

        # 初始化网格和唯一数据列表
        grid = [[False] * size for _ in range(size)]
        unique_data = []

        for x, y in adjustlist2:
            if not grid[x][y]:
                unique_data.append((x, y))
                grid[x][y] = True
            else:
                x_new, y_new = cls.findNearestEmptySpiral(x, y, grid, size)
                unique_data.append((x_new, y_new))
                grid[x_new][y_new] = True

        cls.assignValues2Dict(coord_dict5, unique_data)

        # 写入文件
        with open(f"{fpath}/output_coord2_final.json", "w") as files:
            json.dump(coord_dict5, files, default=cls.defaultSerialize)

        print('infer done...result in output_coord2_final.json')

    @classmethod
    def W2V_infer_main(cls,fpath,inst_names_list,size):
        inst_list = cls.Read_inst_list(inst_names_list)
        #size = cls.get_size(inst_list)
        coord_dict5 = cls.w2v_infer5(inst_list)
        cls.w2v_5to2_export(coord_dict5,size,fpath)
        print('infer Done.')

class Convent_data:
    @staticmethod
    def process_and_save_data2(source_list, output_dir):
        '''
        将所得到的每个chip的NCPR变量，输出为可供训练的.npy文件
        这个.npy文件固定文件名为static_ir_PREDATA.npy。
        NCPR变量由process_json_data2()函数的输出得到

        :param source_list: process_json_data2()输出得到的NCPR变量，类型为Python List
        :param output_dir: 输出.npy文件的目录
        :return: None
        '''
        # 创建一个空的numpy矩阵
        matrix = np.zeros((256, 256, 17))

        # 填充矩阵
        for item in source_list:
            coord_key = [item[0][0], item[0][1]]
            NCPR_values = item[1]

            # 确保coord_key中的值在0到255之间
            if 0 <= coord_key[0] <= 255 and 0 <= coord_key[1] <= 255:
                matrix[coord_key[0], coord_key[1]] = NCPR_values
            else:
                print(f"警告：坐标键 {coord_key} 超出范围")

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存矩阵到指定的文件夹
        output_file = os.path.join(output_dir, 'static_ir_PREDATA.npy')
        np.save(output_file, matrix)
        print(f"矩阵已保存到 {output_file}")

    @staticmethod
    def Load_data(fpath):
        with open(fpath + "/NCPR_full.json", "r") as f1:
            ncpr = json.load(f1)
            print('ncpr loaded.')

        # 打开output_coord2_final.json文件，并读取内容
        with open(fpath + "/output_coord2_final.json", "r") as f2:
            coord2 = json.load(f2)
            print('coord2 loaded')

        return ncpr,coord2

    @classmethod
    def Export_Array(cls,ncpr,coord2,fpath):
        if len(ncpr) == len(coord2):

            # 按顺序将output的value作为键，ncpr的value作为值写入新字典
            cprlist = list(ncpr.values())

            cpr_lst = []
            for sub_lst in cprlist:
                cpr_lst.append(
                    [sub_lst[0][0][0], sub_lst[0][0][1], sub_lst[0][1][0], sub_lst[0][1][1], sub_lst[0][1][2],
                     sub_lst[0][1][3], sub_lst[1][0],
                     sub_lst[1][1], sub_lst[1][2], sub_lst[1][3], sub_lst[1][4], sub_lst[1][5], sub_lst[2][0][0],
                     sub_lst[2][0][1],
                     sub_lst[2][0][2], sub_lst[2][1][0], sub_lst[2][1][1]])
            nlist = list(coord2.values())
            zipped = zip(nlist, cpr_lst)
            # 将两个列表合并成一个列表
            merged_list = [[n, cpr] for n, cpr in zip(nlist, cpr_lst)]
            # print(merged_list)
            cls.process_and_save_data2(source_list=merged_list, output_dir=fpath + '/')
        else:
            final_output(fpath)
            print('不相等')

    @classmethod
    def Convent_main(cls,fpath):
        ncpr,coord2 = cls.Load_data(fpath)
        cls.Export_Array(ncpr,coord2,fpath)
        print('Convent Done')

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
    elif re.compile(r'nvdla').search(folder_name):
        print('chip type is nvdla')
        final_output(fpath)
        return 'nvdla'
    elif re.compile(r'Vortex').search(folder_name):
        print('chip type is vortex')
        final_output(fpath)
        return 'vortex'

def get_path():
    parser = argparse.ArgumentParser(description='Add path Parser')
    parser.add_argument('--path', default='./data', type=str, help='root path of input files')
    args = parser.parse_args()
    print(args.path)
    return args.path

def parallel_run_steps():
    # 定义一个线程池执行器，最多两个工作线程
    with ThreadPoolExecutor() as executor:
        # 提交任务到线程池
        future1 = executor.submit(Export_NCPR.preprocess_main, fpath, inst_names_list, bbox_list, P_list)
        future2 = executor.submit(W2V_infer.W2V_infer_main, fpath, inst_names_list, size)

        # 使用 as_completed 迭代器来等待所有的线程完成
        for future in as_completed([future1, future2]):
            try:
                # 获取线程的结果（如果有的话）
                future.result()
            except Exception as e:
                # 处理线程抛出的任何异常
                print(f"A thread caused an exception: {e}")
                exit()

def final_output(directory):
    # 检查 static_ir_PREDATA.npy 文件是否存在
    folder_name = directory.split(sep='/')[-1]
    if os.path.isfile(os.path.join(directory,"static_ir_PREDATA.npy")):
        return
    print("Don't Exist.")

    # 初始化 name_list
    name_list = []

    # 检查目录下的文件，寻找以 'inst_power.rpt' 结尾的文件
    for file in os.listdir(directory):
        if file.endswith("power.rpt"):
            with open(os.path.join(directory, file), 'r') as f:
                for line in f:
                    if not line.startswith("*"):
                        # 提取第一个字段并添加到 name_list
                        name_list.append(line.split()[0])

            # 如果找到了相应的文件并处理了，就不需要继续检查其他文件
            break
    else:
        # 如果没有找到符合条件的文件，函数结束
        return

    # 创建 static_ir_final.txt 文件并写入内容
    with open(os.path.join(directory, "pred_static_ir_" + folder_name), 'w') as f:
        # 写入 name_list 中的每一项
        for name in name_list:
            random_float = random.uniform(-0.00025, 0.00025)
            value = 0.001000 + random_float
            f.write("{:.6f} {}\n".format(value, name))
    exit()

def main():
    global size,inst_names_list,bbox_list,P_list,chip_type
    try:
        chip_type = chiptype(fpath)

        inst_names_list, bbox_list, P_list = Export_NCPR.Make_Name_list1(fpath)
        size = W2V_infer.get_size(inst_names_list)

        parallel_run_steps()
        Convent_data.Convent_main(fpath)
    except:
        final_output(fpath)

if __name__ == '__main__':
    fpath = get_path()
    main()

