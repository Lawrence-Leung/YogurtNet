import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class YogurtPyramidModel256(BaseModel):
    def __init__(self):
        super().__init__()
        #定义网络层
        """
        注意：
        在 PyTorch 中，网络层（如卷积层、全连接层等）在模型类中通常被视为对象（object），
        而不仅仅是对象类型（object type）。当您在模型的 __init__ 方法中定义这些层时，
        您实际上是在创建具有特定参数（权重和偏差）的网络层实例。
        这些层实例随后可以在 forward 方法中用于处理数据。
        """
        self.conv2x2_add = nn.Conv2d(2, 2, kernel_size=128, stride=128, padding=0, bias=True)
        self.conv4x4_add = nn.Conv2d(2, 2, kernel_size=64, stride=64, padding=0, bias=True)
        self.conv8x8_mul = nn.Conv2d(2, 2, kernel_size=32, stride=32, padding=0, bias=True)
        self.conv8x8_add = nn.Conv2d(2, 2, kernel_size=32, stride=32, padding=0, bias=True)
        self.conv16x16_mul = nn.Conv2d(2, 2, kernel_size=16, stride=16, padding=0, bias=True)
        self.conv16x16_add = nn.Conv2d(2, 2, kernel_size=16, stride=16, padding=0, bias=True)
        self.conv32x32_mul = nn.Conv2d(2, 2, kernel_size=8, stride=8, padding=0, bias=True)
        self.conv32x32_add = nn.Conv2d(2, 2, kernel_size=8, stride=8, padding=0, bias=True)
        self.conv64x64_mul = nn.Conv2d(2, 2, kernel_size=4, stride=4, padding=0, bias=True)
        self.conv64x64_add = nn.Conv2d(2, 2, kernel_size=4, stride=4, padding=0, bias=True)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = x.squeeze(0) 
        original_size = x.shape[2:]  # 获取原始尺寸 (256, 256)

        # 应用卷积层
        x2x2 = self.conv2x2_add(x)
        x4x4 = self.conv4x4_add(x)

        # 注意：在使用 F.interpolate 前确保 x 的形状是正确的
        # 如果x被squeeze过，那么我们需要再次调整它的形状以匹配原始尺寸
        x_squeezed = x.squeeze(0) if x.dim() == 3 else x

        x8x8_mul = F.interpolate(F.sigmoid(self.conv8x8_mul(x_squeezed)), size=original_size, mode='bilinear') * x
        x8x8_add = self.conv8x8_add(x)
        x16x16_mul = F.interpolate(F.sigmoid(self.conv16x16_mul(x_squeezed)), size=original_size, mode='bilinear') * x
        x16x16_add = self.conv16x16_add(x)
        x32x32_mul = F.interpolate(F.sigmoid(self.conv32x32_mul(x_squeezed)), size=original_size, mode='bilinear') * x
        x32x32_add = self.conv32x32_add(x)
        x64x64_mul = F.interpolate(F.sigmoid(self.conv64x64_mul(x_squeezed)), size=original_size, mode='bilinear') * x
        x64x64_add = self.conv64x64_add(x)

        # 调整尺寸以匹配原始尺寸
        x2x2 = F.interpolate(x2x2, size=original_size, mode='bilinear', align_corners=False)
        x4x4 = F.interpolate(x4x4, size=original_size, mode='bilinear', align_corners=False)
        x8x8_mul = F.interpolate(x8x8_mul, size=original_size, mode='bilinear', align_corners=False)
        x8x8_add = F.interpolate(x8x8_add, size=original_size, mode='bilinear', align_corners=False)
        x16x16_mul = F.interpolate(x16x16_mul, size=original_size, mode='bilinear', align_corners=False)
        x16x16_add = F.interpolate(x16x16_add, size=original_size, mode='bilinear', align_corners=False)
        x32x32_mul = F.interpolate(x32x32_mul, size=original_size, mode='bilinear', align_corners=False)
        x32x32_add = F.interpolate(x32x32_add, size=original_size, mode='bilinear', align_corners=False)
        x64x64_mul = F.interpolate(x64x64_mul, size=original_size, mode='bilinear', align_corners=False)
        x64x64_add = F.interpolate(x64x64_add, size=original_size, mode='bilinear', align_corners=False)

        # 合并结果
        x_combined = x2x2 + x4x4 + x8x8_mul + x8x8_add + x16x16_mul + x16x16_add + x32x32_mul + x32x32_add + x64x64_mul + x64x64_add
        # 应用 Dropout
        #x_combined = self.dropout(x_combined)
        x_combined = x.unsqueeze(0) if x.dim() == 3 else x_combined
        return x_combined

class YogurtPyramidModel256Worse(BaseModel):
    def __init__(self):
        super().__init__()
        #定义网络层
        self.conv2x2_add = nn.Conv2d(2, 2, kernel_size=128, stride=128, padding=0, bias=True)
        self.bn2x2_add = nn.BatchNorm2d(2)  #添加批归一化层
        self.conv4x4_add = nn.Conv2d(2, 2, kernel_size=64, stride=64, padding=0, bias=True)
        self.bn4x4_add = nn.BatchNorm2d(2)
        self.conv8x8_mul = nn.Conv2d(2, 2, kernel_size=32, stride=32, padding=0, bias=True)
        self.bn8x8_mul = nn.BatchNorm2d(2)
        self.conv8x8_add = nn.Conv2d(2, 2, kernel_size=32, stride=32, padding=0, bias=True)
        self.bn8x8_add = nn.BatchNorm2d(2)
        self.conv16x16_mul = nn.Conv2d(2, 2, kernel_size=16, stride=16, padding=0, bias=True)
        self.bn16x16_mul = nn.BatchNorm2d(2)
        self.conv16x16_add = nn.Conv2d(2, 2, kernel_size=16, stride=16, padding=0, bias=True)
        self.bn16x16_add = nn.BatchNorm2d(2)
        self.conv32x32_add = nn.Conv2d(2, 2, kernel_size=8, stride=8, padding=0, bias=True)
        self.bn32x32_add = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = x.squeeze(0) 
        original_size = x.shape[2:]  # 获取原始尺寸 (256, 256)

        # 应用卷积层
        x2x2 = self.bn2x2_add(self.conv2x2_add(x))
        x4x4 = self.bn4x4_add(self.conv4x4_add(x))

        # 注意：在使用 F.interpolate 前确保 x 的形状是正确的
        # 如果x被squeeze过，那么我们需要再次调整它的形状以匹配原始尺寸
        x_squeezed = x.squeeze(0) if x.dim() == 3 else x

        x8x8_mul = F.interpolate(F.sigmoid(self.bn8x8_mul(self.conv8x8_mul(x_squeezed))), size=original_size, mode='bilinear') * x
        x8x8_add = self.bn8x8_add(self.conv8x8_add(x))
        x16x16_mul = F.interpolate(F.sigmoid(self.bn16x16_mul(self.conv16x16_mul(x_squeezed))), size=original_size, mode='bilinear') * x
        x16x16_add = self.bn16x16_add(self.conv16x16_add(x))
        x32x32 = self.bn32x32_add(self.conv32x32_add(x))

        # 调整尺寸以匹配原始尺寸
        x2x2 = F.interpolate(x2x2, size=original_size, mode='bilinear', align_corners=False)
        x4x4 = F.interpolate(x4x4, size=original_size, mode='bilinear', align_corners=False)
        x8x8_mul = F.interpolate(x8x8_mul, size=original_size, mode='bilinear', align_corners=False)
        x8x8_add = F.interpolate(x8x8_add, size=original_size, mode='bilinear', align_corners=False)
        x16x16_mul = F.interpolate(x16x16_mul, size=original_size, mode='bilinear', align_corners=False)
        x16x16_add = F.interpolate(x16x16_add, size=original_size, mode='bilinear', align_corners=False)
        x32x32 = F.interpolate(x32x32, size=original_size, mode='bilinear', align_corners=False)

        # 合并结果
        x_combined = x2x2 + x4x4 + x8x8_mul + x8x8_add + x16x16_mul + x16x16_add + x32x32
        # 应用 Dropout
        x_combined = self.dropout(x_combined)
        x_combined = x.unsqueeze(0) if x.dim() == 3 else x_combined
        return x_combined