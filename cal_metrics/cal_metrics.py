import numpy as np
import matplotlib.pyplot as plt

def calculate_mae(golden, predict):
    absolute_errors = np.abs(golden - predict)
    mae = np.mean(absolute_errors)
    return mae

def calculate_average_mae(goldens, predicts):
    mae_values = []
    for i in range(len(predicts)):
        mae = calculate_mae(np.array(goldens[i]), np.array(predicts[i]))
        mae_values.append(mae)
    average_mae = np.mean(mae_values)
    return average_mae

def calculate_cc(golden, predict):
    cc = np.corrcoef(golden, predict)[0, 1]
    return cc

def calculate_average_cc(goldens, predicts):
    cc_values = []
    for i in range(len(predicts)):
        cc = calculate_cc(np.array(goldens[i]), np.array(predicts[i]))
        cc_values.append(cc)
    average_cc = np.mean(cc_values)
    return average_cc

def cal_metrics(goldens, predicts, plot):
    average_mae = calculate_average_mae(np.array(goldens), np.array(predicts))
    average_cc = calculate_average_cc(np.array(goldens), np.array(predicts))
    t = f'CC = {average_cc:.4f}\nMAE = {average_mae:.4f}'

    if plot:  # 以图片形式输出均值metrics，默认关闭，在main中让plot=True以开启。
        max_goldens = np.max(goldens, axis = None)
        g_max = np.max(max_goldens)
        max_predicts = np.max(predicts, axis = None)
        p_max = np.max(max_predicts)
        mm = max(g_max, p_max)
        thres = 0.05
        x1 = [0, mm]
        y1 = [0, mm]
        y2 = [0, mm * (1-thres) ]
        y3 = [0, mm * (1+thres) ]
        plt.plot(x1, y1, color='grey', lw=1, ls = '--', label='y=x')
        plt.plot(x1, y2, color='g', lw=1, label=f'{int(100*thres)}% diff', alpha=0.8)
        plt.plot(x1, y3, color='g', lw=1, alpha=0.8)
        ps = np.array(predicts)
        gs = np.array(goldens)
        for ii in range(len(ps)):
            plt.scatter(gs[ii], ps[ii], color = 'b', marker='o', edgecolor='none', s=12, alpha=0.6)
        plt.xlabel('golden_ir_drop(mV)')
        plt.ylabel('predict_ir_drop(mV)')
        bbox = dict(ec = 'black', lw =1, facecolor = 'white', alpha = 0.6)
        plt.text(0.03, 0.9, t, size = 10, color = 'black', transform = plt.gca().transAxes, bbox = bbox)
        out_fig = f'ir_diff.png'
        out_fig = get_file_path(out_fig)
        plt.title('IR Drop comparison')
        plt.legend(loc = 'lower right')
        plt.savefig(out_fig)
        plt.show()
    else:
        print(t)