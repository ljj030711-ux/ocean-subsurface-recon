"""
涡旋反演模型模块
包含 CNN 和其他神经网络架构
"""

import torch
import torch.nn as nn


class EddyAwareCNN(nn.Module):
    """
    涡旋感知 CNN 模型（MVP版本）
    
    输入：
        - 5 通道：[SST, SSS, SSH, EKE, ∇SSH]
    
    输出：
        - 1 通道：水下温度反演结果
    
    架构：简洁高效的 4 层卷积 + ReLU
    """
    
    def __init__(self, in_channels=5, out_channels=1, depth_multiplier=1):
        """
        初始化模型
        
        Args:
            in_channels: 输入通道数 (默认5: sst, sss, ssh, eke, grad)
            out_channels: 输出通道数 (默认1)
            depth_multiplier: 深度倍数（用于控制模型大小）
        """
        super().__init__()
        
        # 基础通道数
        ch = 32 * depth_multiplier
        
        self.net = nn.Sequential(
            # Conv 1: in_channels → 32
            nn.Conv2d(in_channels, ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            
            # Conv 2: 32 → 64
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            
            # Conv 3: 64 → 32
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            
            # Conv 4: 32 → 1 (输出层，不使用激活函数)
            nn.Conv2d(ch, out_channels, kernel_size=1, bias=True),
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """使用 Kaiming 初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
        
        Returns:
            output: 输出张量 [B, 1, H, W]
        """
        return self.net(x)


class EddyUNet(nn.Module):
    """
    涡旋反演 UNet 模型（升级版本）
    
    包含编码器-解码器结构，更强的特征提取能力
    """
    
    def __init__(self, in_channels=5, out_channels=1):
        """
        初始化 UNet 模型
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        
        # 编码器
        self.encoder1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 中间层
        self.bridge = self._conv_block(64, 128)
        
        # 解码器
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(64, 32)
        
        # 输出层
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        """卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """前向传播"""
        # 编码
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        
        # 中间
        bridge = self.bridge(pool2)
        
        # 解码
        upconv2 = self.upconv2(bridge)
        dec2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        upconv1 = self.upconv1(dec2)
        dec1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # 输出
        output = self.final(dec1)
        
        return output


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        """初始化残差块"""
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """前向传播"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out


class EddyResNet(nn.Module):
    """
    涡旋反演 ResNet 模型
    使用残差连接改善深层网络训练
    """
    
    def __init__(self, in_channels=5, out_channels=1):
        """初始化 ResNet 模型"""
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, 
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建残差层"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.upconv2(x)
        x = self.upconv1(x)
        x = self.final(x)
        
        return x
