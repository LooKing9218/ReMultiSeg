import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layers import unetConv2,unetUp,unetConv2_dilation,unetUp_cat
from models.utils.init_weights import init_weights
import math

class UNet_2(nn.Module):

    def __init__(self, in_channels=1,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.eca = eca_layer(filters[4])
        self.spp = SPPblock(filters[4])

        # upsampling
        self.up_concat4 = unetUp_cat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)       # 32*512*512
        maxpool1 = self.maxpool1(conv1)  # 32*256*256

        conv2 = self.conv2(maxpool1)     # 64*256*256
        maxpool2 = self.maxpool2(conv2)  # 64*128*128

        conv3 = self.conv3(maxpool2)     # 128*128*128
        maxpool3 = self.maxpool3(conv3)  # 128*64*64

        conv4 = self.conv4(maxpool3)     # 256*64*64
        maxpool4 = self.maxpool4(conv4)  # 256*32*32

        center = self.center(maxpool4)   # 512*32*32
        eca = self.eca(center)             # 512*32*32
        spp = self.spp(center)               # 4*32*64
        pe = torch.cat([eca, spp], dim=1)


        up4 = self.up_concat4(pe,conv4)  # 128*64*128
        up3 = self.up_concat3(up4,conv3)     # 64*128*256
        up2 = self.up_concat2(up3,conv2)     # 32*256*512
        up1 = self.up_concat1(up2,conv1)     # 16*512*1024

        final_1 = self.final_1(up1)

        return torch.sigmoid(final_1)

class UNet_2_2(nn.Module):

        def __init__(self, in_channels=1, n_classes=4, feature_scale=4, is_deconv=True, is_batchnorm=True):
            super(UNet_2_2, self).__init__()
            self.is_deconv = is_deconv
            self.in_channels = in_channels
            self.is_batchnorm = is_batchnorm
            self.feature_scale = feature_scale

            filters = [64, 128, 256, 512, 1024]
            filters = [int(x / self.feature_scale) for x in filters]

            # downsampling
            self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)

            self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
            self.eca = eca_layer(filters[4])
            self.spp = SPPblock(filters[4])

            # upsampling
            self.up_concat4 = unetUp_cat(filters[4], filters[3], self.is_deconv)
            self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
            self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
            self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

            # final conv (without any concat)
            self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
            # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
            # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)

            # initialise weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init_weights(m, init_type='kaiming')
                elif isinstance(m, nn.BatchNorm2d):
                    init_weights(m, init_type='kaiming')

        def forward(self, inputs):
            conv1 = self.conv1(inputs)  # 32*512*512
            maxpool1 = self.maxpool1(conv1)  # 32*256*256

            conv2 = self.conv2(maxpool1)  # 64*256*256
            maxpool2 = self.maxpool2(conv2)  # 64*128*128

            conv3 = self.conv3(maxpool2)  # 128*128*128
            maxpool3 = self.maxpool3(conv3)  # 128*64*64

            conv4 = self.conv4(maxpool3)  # 256*64*64
            maxpool4 = self.maxpool4(conv4)  # 256*32*32

            center = self.center(maxpool4)  # 512*32*32
            #eca = self.eca(center)  # 512*32*32
            spp = self.spp(center)  # 4*32*64
            pe = torch.cat([center, spp], dim=1)

            up4 = self.up_concat4(pe, conv4)  # 128*64*128
            up3 = self.up_concat3(up4, conv3)  # 64*128*256
            up2 = self.up_concat2(up3, conv2)  # 32*256*512
            up1 = self.up_concat1(up2, conv1)  # 16*512*1024

            final_1 = self.final_1(up1)

            return F.log_softmax(final_1, dim=1)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[1, 1], stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.fcw = nn.Linear(4, 4, bias=True).cuda()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, b, h, w = x.size(1), x.size(0), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        #self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        out1 = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4], 1)
        w1, w2, w3, w6 = out1.split([1,1,1,1], dim=1)
        w1, w2, w3, w6 =w1.view(4,-1).transpose(0,1), w2.view(4,-1).transpose(0,1), w3.view(4,-1).transpose(0,1), w6.view(4,-1).transpose(0,1)
        o1, o2, o3, o6 = self.fcw(w1).transpose(0,1).reshape(4, h, w).unsqueeze(0), self.fcw(w2).transpose(0,1).reshape(4, h, w).unsqueeze(0), self.fcw(w3).transpose(0,1).reshape(4, h, w).unsqueeze(0), self.fcw(w6).transpose(0,1).reshape(4, h, w).unsqueeze(0)

        out = torch.cat([o1, o2, o3, o6 ], dim = 0)

        #out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = UNet_1(in_channels=1, n_classes=4, is_deconv=True).cuda()
    print(net)
    x = torch.rand((4, 1, 256, 128)).cuda()
    forward = net.forward(x)
    print(forward)
    print(type(forward))