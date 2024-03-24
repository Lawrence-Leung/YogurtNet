import os
import sys
import subprocess
import time

import argparse

parser = argparse.ArgumentParser(description='Add path Parser')
parser.add_argument('--entrance',default='.', type=str, help='root path of files entrance')
parser.add_argument('--pyscript',default='', type=str, help='path of pyscript')
args = parser.parse_args()
# 定义输入、输出文件路径
entrance1 = args.entrance
pyentrance =args.pyscript

def find_directories(root_path, filename):
    directories = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if filename in file:  # This line checks if the filename substring is in each file
                #full_path = os.path.join(root, 'static_ir_PREDATA.npy')
                #if not os.path.exists(full_path):
                directories.append(root)
                break  # Prevent adding the same directory multiple times
    print(directories)
    return directories


def run_script(script_path, directories):
    """
    对每个找到的目录运行指定的Python脚本。
    """
    n = 0
    count = 0
    skip = 0
    for directory in directories:
        #time.sleep(1)
        #if os.path.exists(directory+'/static_ir_PREDATA.npy'):
        #    skip+=1
        #    print('已经存在,skip',skip)
        #    continue
        #else:
        if len(directories) == 1:
            arg = ['--path=' + directory]
            subprocess.run(["python", script_path] + arg)
            break

        elif n < process_num:
            count += 1
            arg = ['--path=' + directory]
            print(f"现在是第{count}个,正在运行脚本 '{script_path}' 使用数据目录: {directory}")
            subprocess.Popen(["python", script_path] + arg)
            n += 1

        elif n == process_num:
            n = 0
            count += 1
            print('Running')
            print(str(count)+'/'+str(len(directories)))
            subprocess.run(["python", script_path] + arg)
            #time.sleep(2)

def main():
    '''
    if len(sys.argv) != 3:
        print("Usage: python script.py <entrance1 directory> <pyentrance.py path>")
        sys.exit(1)

    entrance1 = sys.argv[1]
    pyentrance = sys.argv[2]
    '''

    if not os.path.isdir(entrance1):
        print(f"Directory '{entrance1}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(pyentrance):
        print(f"Python script '{pyentrance}' does not exist.")
        sys.exit(1)

    print(f"开始搜索 '{entrance1}' 以找到包含 'pulpino_top.inst.power.rpt' 的目录")
    directories = find_directories(entrance1, "power.rpt")

    if directories:
        print("开始运行脚本...")
        run_script(pyentrance, directories)
    else:
        print("没有找到包含指定文件的目录。")

if __name__ == "__main__":
    process_num = 30
    main()
    #lis = find_directories(entrance1, "pulpino_top.inst.power.rpt")

    #print(lis)
    #print(len(lis))


# python autoprocess.py --entrance=/path/to/file
