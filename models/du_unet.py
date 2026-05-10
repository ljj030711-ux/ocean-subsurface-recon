"""
Du_Unet 双分支反演模型
高分辨率 SST 分支提取细尺度特征，低分辨率 SSH/SSS 分支提取多参数特征，融合后输出单层温度或盐度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Du_Unet(nn.Module):
    """双分支 2D->2D UNet。"""

    def __init__(self, out_channels=1, sst_channels=1, surface_channels=2, base_channels=32):
        super().__init__()
        ch = int(base_channels)

        self.sst_enc1 = self._conv_block(sst_channels, ch)
        self.sst_pool1 = nn.MaxPool2d(2)
        self.sst_enc2 = self._conv_block(ch, ch * 2)
        self.sst_pool2 = nn.MaxPool2d(2)
        self.sst_enc3 = self._conv_block(ch * 2, ch * 4)
        self.sst_pool3 = nn.MaxPool2d(2)
        self.sst_enc4 = self._conv_block(ch * 4, ch * 4)
        self.sst_pool4 = nn.MaxPool2d(2)
        self.sst_bridge = self._conv_block(ch * 4, ch * 8)
        self.sst_up4 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.sst_dec4 = self._conv_block(ch * 8, ch * 4)
        self.sst_up3 = nn.ConvTranspose2d(ch * 4, ch * 4, kernel_size=2, stride=2)
        self.sst_dec3 = self._conv_block(ch * 8, ch * 4)
        self.sst_up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.sst_dec2 = self._conv_block(ch * 4, ch * 2)
        self.sst_up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.sst_dec1 = self._conv_block(ch * 2, ch)
        self.sst_to_low = self._conv_block(ch, ch)

        self.surface_enc = self._conv_block(surface_channels, ch)
        self.surface_pool = nn.MaxPool2d(2)
        self.surface_bridge = self._conv_block(ch, ch * 2)
        self.surface_up = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.surface_dec = self._conv_block(ch * 2, ch)

        self.fusion = nn.Sequential(
            self._conv_block(ch * 2, ch * 2),
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channels, kernel_size=1),
        )
        self._init_weights()

    def _conv_block(self, in_ch, out_ch):
        """卷积、归一化和激活的基础块。"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        """初始化卷积和归一化层。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _sst_branch(self, sst, out_size):
        enc1 = self.sst_enc1(sst)
        enc2 = self.sst_enc2(self.sst_pool1(enc1))
        enc3 = self.sst_enc3(self.sst_pool2(enc2))
        enc4 = self.sst_enc4(self.sst_pool3(enc3))
        bridge = self.sst_bridge(self.sst_pool4(enc4))

        up4 = self._match_size(self.sst_up4(bridge), enc4)
        dec4 = self.sst_dec4(torch.cat([up4, enc4], dim=1))
        up3 = self._match_size(self.sst_up3(dec4), enc3)
        dec3 = self.sst_dec3(torch.cat([up3, enc3], dim=1))
        up2 = self._match_size(self.sst_up2(dec3), enc2)
        dec2 = self.sst_dec2(torch.cat([up2, enc2], dim=1))
        up1 = self._match_size(self.sst_up1(dec2), enc1)
        dec1 = self.sst_dec1(torch.cat([up1, enc1], dim=1))

        low = F.interpolate(dec1, size=out_size, mode="bilinear", align_corners=False)
        return self.sst_to_low(low)

    def _surface_branch(self, ssh_sss):
        enc = self.surface_enc(ssh_sss)
        bridge = self.surface_bridge(self.surface_pool(enc))
        up = self._match_size(self.surface_up(bridge), enc)
        return self.surface_dec(torch.cat([up, enc], dim=1))

    def forward(self, sst, ssh_sss):
        """
        Args:
            sst: (B,1,160,160)
            ssh_sss: (B,2,64,64)，通道顺序为 SSH、SSS
        """
        sst_feat = self._sst_branch(sst, ssh_sss.shape[-2:])
        surface_feat = self._surface_branch(ssh_sss)
        return self.fusion(torch.cat([sst_feat, surface_feat], dim=1))
