import numpy as np
import torch
import torch.nn as nn

class FruitClassificationModel(nn.Module):
    def __init__(self, 
                 num_conv_layers=3,          # 自定义卷积层数
                 initial_channels=16,        # 初始卷积通道数
                 channel_multiplier=2,       # 通道数倍增因子
                 input_size=(3, 224, 224),   # 输入图像尺寸 (C, H, W)
                 num_classes=100 # 水果类别数
                ):           
        super(FruitClassificationModel, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.num_classes = num_classes
        
        # 构建卷积层
        conv_layers = []
        in_channels = input_size[0]  # 输入通道数，RGB图像为3
        out_channels = initial_channels
        
        for _ in range(num_conv_layers):
            # 卷积层 + 批归一化 + ReLU + 最大池化
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            conv_layers.append(conv_block)
            
            # 更新输入输出通道数
            in_channels = out_channels
            out_channels = min(out_channels * channel_multiplier, 512)  # 限制最大通道数
        
        self.features = nn.Sequential(*conv_layers)
        
        # 计算卷积层输出特征图的大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self.features(dummy_input)
            self.feature_size = dummy_output.view(1, -1).size(1)
        
        # 全连接层分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x

    