""" Full assembly of the parts to form the complete network """

from models.unet_r3.unet_parts import *
from models.block import DWConv2d, DWConv2dReLU, ChannelAligner

# log/synapse/UNetR3_ex1_bilinear 82.19
class UNetR3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetR3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        self.up_4_3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2dReLU(1024, 512),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2dReLU(512, 256)
        )

        self.up_3_2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2dReLU(512, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2dReLU(256, 128)
        )

        self.up_2_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2dReLU(256, 128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2dReLU(128, 64)
        )


        self.seghead = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            DWConv2d(128, n_classes)
        )
        


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # x1 (64, 224, 224)
        # x2 (128, 112, 112)
        # x3 (256, 56, 56)
        # x4 (512, 28, 28)
        # x5 (1024, 14, 14)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        y3 = x3 + self.up_4_3(x5)
        y3 = self.down3(y3)

        y2 = x2 + self.up_3_2(y3)
        y2 = self.down2(y2)

        y1 = x1 + self.up_2_1(y2)
        y1 = self.down1(y1)

        y = self.seghead(y1)

        return y

    def use_checkpointing(self):
        pass

    def flops(self):
        from models.unet.utils import print_flops_params
        print_flops_params(self)


# class UNetR3(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNetR3, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(512, 1024 // factor))
        
#         self.up_4_3 = nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#             ChannelAligner(1024, 512),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             ChannelAligner(512, 256),
#         )

#         self.up_3_2 = nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#             ChannelAligner(512, 256),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             ChannelAligner(256, 128),
#         )

#         self.up_2_1 = nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#             ChannelAligner(256, 128),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             ChannelAligner(128, 64),
#         )

#         self.seghead = nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#             DWConv2d(128, n_classes)
#         )
        


#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
#         # x1 (64, 224, 224)
#         # x2 (128, 112, 112)
#         # x3 (256, 56, 56)
#         # x4 (512, 28, 28)
#         # x5 (1024, 14, 14)

#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         y3 = x3 + self.up_4_3(x5)
#         y3 = self.down3(y3)

#         y2 = x2 + self.up_3_2(y3)
#         y2 = self.down2(y2)

#         y1 = x1 + self.up_2_1(y2)
#         y1 = self.down1(y1)

#         y = self.seghead(y1)

#         return y

#     def use_checkpointing(self):
#         pass

#     def flops(self):
#         from models.unet.utils import print_flops_params
#         print_flops_params(self)