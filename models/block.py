import torch
import torch.nn as nn


class DWConv2d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DWConv2d, self).__init__()
        # 也相当于分组为1的分组卷积
        # self.depth_conv = nn.Conv2d(in_channels=in_ch,
        #                             out_channels=in_ch,
        #                             kernel_size=3,
        #                             stride=1,
        #                             padding=1,
        #                             groups=in_ch,
        #                             bias=False
        #                             )
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)
    def forward(self,input):
        # out = self.depth_conv(input)
        out = self.point_conv(input)
        return out
    

class DWConv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        conv = DWConv2d(
            in_channels,
            out_channels,
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(DWConv2dReLU, self).__init__(conv, bn, relu)


class ChannelAligner(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ChannelAligner, self).__init__()
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self,input):
        out = self.relu(self.bn(self.point_conv(input)))
        return out
    

class CNN2SwinFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C)
        return x
    
class Swin2CNNFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, L, C = x.shape
        W, H = int(L**0.5), int(L**0.5)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, W, H)
        return x