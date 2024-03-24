import torch.nn.functional as F


def nll_loss(output, target):
    "不适合使用"
    return F.nll_loss(output, target)

"""
适合回归任务的损失函数
"""
def L1Loss(output, target):
    '''
    平均绝对误差损失（Mean Absolute Error Loss, MAE）
    计算模型输出和目标之间的平均绝对值差。适用于当您希望所有的误差都同等重要时。
    '''
    return F.l1_loss(output, target)

def SmoothL1Loss(output, target):
    '''
    平滑L1损失（Smooth L1 Loss）
    描述：结合了MSE损失和MAE损失的特点，对于较大的误差采用L1损失，对于较小的误差采用L2损失。这使得损失函数对异常值不那么敏感，同时在接近正确解时更平滑。
    公式：平滑L1损失在不同的误差范围内使用不同的公式。
    '''
    return nn.smooth_l1_loss(output, target)

def MSELoss(output, target):
    '''
    均方误差损失（Mean Squared Error Loss, MSE）
    计算模型输出和目标之间的均方差。适用于当您希望小误差被放大处理，强调更精确的拟合。
    '''
    return F.mse_loss(output, target)