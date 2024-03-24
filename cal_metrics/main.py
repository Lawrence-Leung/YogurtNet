
import os, gzip
import argparse
from multiprocessing.pool import Pool
from cal_metrics import cal_metrics


class Paraser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--golden_path', default='./data', help='golden data的文件夹')
        self.parser.add_argument('--pred_path', default='./pred_static_ir_report', help='pred data的文件夹')
        self.parser.add_argument('--plot', default=False, help='画出计算结果')
        self.parser.add_argument('--n_subprocess', default=16, help='多进程数量')


def diff_detail(v1, v2):
    reletive = 0
    if v1 == 0.0 and v2 == 0.0:
        reletive = 0
    elif v1 == 0.0 or v2 == 0.0:
        reletive = 100.0
    else:
        reletive = abs(v1-v2)/v1*100
    absolute = v1 - v2
    return [v1, v2, absolute, reletive]

def get_data(data_path):
    data = {}
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
    else:
        files = [os.path.basename(data_path)]

    for i, file in enumerate(files):
        if file.startswith('static_ir'):  #获取golden data
            case_name = os.path.basename(data_path)
            ir_rpt_path = os.path.join(data_path, file)
            if file.endswith('.gz'):
                fp = gzip.open(ir_rpt_path, 'rt')
            else:
                fp = open(ir_rpt_path, 'r')
            for line in fp:
                if line.startswith('#'):
                    continue
                lline = line.split()
                nn1 = float(lline[1])
                nn2 = float(lline[2])
                drop = (nn1 + nn2)* 1e3
                inst = lline[-1]
                data[case_name + '/' + inst] = drop
            fp.close()
            break
        elif file.startswith('pred_static_ir_'): #获取prediction data
            case_name = file.split('pred_static_ir_')[1]
            if file.endswith('.gz'):
                fp = gzip.open(data_path, 'rt')
                case_name = case_name.split('.gz')[0]
            else:
                fp = open(data_path, 'r')
            for line in fp:
                if line.startswith('#'):
                    continue
                lline = line.split()
                drop = (float(lline[0]))* 1e3
                inst = lline[1]
                data[case_name + '/' + inst] = drop
            fp.close()
        elif i == len(files) - 1:
            raise ValueError("路径{}下没有static_ir report文件".format(data_path))
    return data

def compare_ir_drop_with_statistics(arg):
    pool = Pool(processes=arg.n_subprocess)
    golden_dirs = sorted(os.listdir(arg.golden_path))
    golden_dirs = [os.path.join(arg.golden_path, i) for i in golden_dirs]
    golden_datas = pool.map(get_data, golden_dirs)
    pred_dirs = sorted(os.listdir(arg.pred_path))
    pred_dirs = [os.path.join(arg.pred_path, i) for i in pred_dirs]
    pred_datas = pool.map(get_data, pred_dirs)

    results = []
    goldens = []
    predicts = []
    for i in range(len(golden_datas)):
        golden_data = golden_datas[i]
        pred_data = pred_datas[i]
        golden = []
        predict = []
        result = {}
        inii=0
        minii=0
        for inst in golden_data:
            v1 = golden_data[inst]
            v2 = 0.0
            if inst not in pred_data: # 将pred_data的instance数量按golden对齐
                inii+=1;
            else:
                v2 = pred_data[inst]
            if v1 < 1.0:             # 滤掉小于1mV的instance
                minii+=1;
                continue
            result[inst] = diff_detail(v1, v2)
            golden.append(v1)
            predict.append(v2)
        goldens.append(golden)
        predicts.append(predict)
        results.append(result)

    cal_metrics(goldens, predicts, arg.plot) # 计算metrics
     

if __name__ == "__main__":
    argp = Paraser()
    arg = argp.parser.parse_args()
    compare_ir_drop_with_statistics(arg)