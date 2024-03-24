import torch


def accuracy(output, target):
    """
    适合分类问题，不适合回归预测问题。
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """
    适合分类问题，不适合回归预测问题。
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def R_squares(output, target):
    """
    R²（R-squared, 决定系数）:
    衡量模型预测值的变异量占总变异量的比例，值越接近1表示模型拟合效果越好。
    公式较复杂，涉及总平方和（TSS）和残差平方和（RSS）。
    """
    ss_total = torch.sum((target - torch.mean(target)) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r_squared = 1 - (ss_res / ss_total)
    return r_squared
