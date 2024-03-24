import os
import json
import numpy as np
import glob
# ThreadPool处理库，转为密集IO型和密集CPU计算型设计，优化性能 :)
from concurrent.futures import ThreadPoolExecutor, as_completed
# 我们采用双重性能并行，
# 外层靠Linux内核，多进程处理；
# 内层靠ThreadPool，IO和CPU靠Concurrent的ThreadPool，GPU靠CUDA

IMG_SIZE = 256

'''
    公用工具函数，命名规则以global结尾
'''
def load_json_global(file_path):
    '''
    加载JSON文件(公用函数)
    :param file_path: 某个文件地址，要求是特定的.json文件
    :return: 加载好的JSON变量
    '''
    """加载JSON文件"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        raise FileNotFoundError(f"Process 6 JSON file not found. {e}")

def delfile(directory):
    for filename in glob.glob(os.path.join(directory, 'static_ir_P*')):
        try:
            os.remove(filename)
            # print(f"Deleted {filename}")
        except OSError as e:
            # print(f"Error: {filename} : {e.strerror}")

    # 删除 'NCPR_full.json' 和 'output_coord2_final.json' 文件
    for specific_file in ['NCPR_full.json', 'output_coord2_final.json']:
        file_path = os.path.join(directory, specific_file)
        try:
            os.remove(file_path)
            # print(f"Deleted {file_path}")
        except OSError as e:
            # print(f"Error: {file_path} : {e.strerror}")

'''
    Process 9 代码，如下所示，命名规则带有数字9
'''
def find_npy_files9(root_dir):
    '''
    递归查找所有的static_ir_POSTDATA.npy文件，并返回包含这些文件的目录
    :param root_dir: 文件目录，即包含所有instance文件的统一入口地址
    :return: 所有包含文件的子目录，这是一个Iterator。
    '''
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('static_ir_POSTDATAfinal.npy'):
                yield root  # 返回包含文件的目录

def process_data9(npy_file, coord_json):
    '''
    处理数据并生成static_ir_final.dat文件
    :param npy_file: 目标.npy文件名
    :param coord_json: 坐标映射json（是json变量，不是json文件）
    :param cvrt_json: 每个static_ir的固定参数所存放的json，如"VDD"之类的（是json变量，不是json文件）
    :return: None
    '''
    """处理数据并生成static_ir_final.dat文件"""
    data = np.load(npy_file)
    static_ir_final = []

    # 反转coord_json的键值对，以便提高性能。
    reversed_coord_json = {str(v): k for k, v in coord_json.items()}

    for name_x in range(IMG_SIZE):
        for name_y in range(IMG_SIZE):
            inst_name_key = str([name_x, name_y])
            if inst_name_key in reversed_coord_json:
                inst_name = reversed_coord_json[inst_name_key].replace("_-", "[").replace("-_", "]")
                vdd_drop, gnd_bounce = data[name_x, name_y, 0], data[name_x, name_y, 1]
                #loc_x, loc_y = data[name_x, name_y, 2], data[name_x, name_y, 3]
                #ideal_vdd = float(cvrt_json["ideal_vdd"])
                #pwr_net = cvrt_json["pwr_net"]
                #inst_vdd = round(ideal_vdd - vdd_drop - gnd_bounce, 4)
                ir_drop = vdd_drop + gnd_bounce
                static_ir_final.append(f"{ir_drop:.6f} {inst_name}")
                #print(
                #    f"{inst_vdd} {vdd_drop:.5f} {gnd_bounce:.6f} {ideal_vdd} {pwr_net} {loc_x:.3f},{loc_y:.3f} {inst_name}")
            else:
                pass
                #print("错误：未找到匹配的键值对")

    # 生成static_ir_final.dat文件
    folder_name = os.path.dirname(npy_file).split(sep='/')[-1]
    final_path = os.path.join(os.path.dirname(npy_file), "pred_static_ir_" + folder_name)
    with open(final_path, 'w') as file:
        #file.write("# inst_vdd vdd_drop gnd_bounce ideal_vdd pwr_net location inst_name\n")
        file.write("\n".join(static_ir_final))

def process_folder9(folder):
    """处理单个文件夹的数据"""
    #print(folder, "Process 9")
    coord_json_path = os.path.join(folder, 'output_coord2_final.json')
    #cvrt_json_path = os.path.join(folder, 'static_ir_PostCvrt.json')
    npy_file_path = os.path.join(folder, 'static_ir_POSTDATAfinal.npy')
    #print(coord_json_path,'\n', npy_file_path)

    if os.path.exists(coord_json_path) and os.path.exists(npy_file_path):#and os.path.exists(cvrt_json_path):
        coord_json = load_json_global(coord_json_path)
        #cvrt_json = load_json_global(cvrt_json_path)
        process_data9(npy_file_path, coord_json)
    else:
        raise FileNotFoundError(f"Process 9 File not found.")

def check_final_files9(root_dir):
    '''
    检查每个子文件夹的static_ir_final.dat文件中的实例数量
    :param root_dir: 入口文件夹
    :return: None
    '''
    for folder in find_npy_files9(root_dir):
        coord_json_path = os.path.join(folder, 'map_table', 'output_coord2_final.json')
        final_dat_path = os.path.join(folder, 'static_ir_final.dat')

        if os.path.exists(coord_json_path) and os.path.exists(final_dat_path):
            with open(coord_json_path, 'r') as f:
                coord_json = json.load(f)
            with open(final_dat_path, 'r') as f:
                lines = f.readlines()[1:]  # 跳过首行
            if len(coord_json) != len(lines):
                print(f"实例数量不匹配：{folder}")
            else:
                print(f"实例数量匹配：{folder}")

def Proc9Main(root_dir):
    '''
    Process 9主函数，用于并行处理各个子文件夹
    :param
           root_dir: 入口文件夹
    :return: None
    '''

    with ThreadPoolExecutor() as executor:
        # 创建一个future到文件夹路径的映射
        future_to_folder = {executor.submit(process_folder9, folder): folder for folder in find_npy_files9(root_dir)}

        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                future.result()
                delfile(folder)
                print(f"处理完成：{folder}/pred_static_ir_xxx")
            except Exception as e:
                raise AttributeError(f"Process 9 Main failed. {e}")



if __name__ == '__main__':
    print("Running")
    Proc9Main('/home/stxianyx/code/eda/data')
    print("Done!")