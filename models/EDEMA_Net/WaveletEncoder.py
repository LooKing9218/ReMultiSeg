import torch
import torch.nn as nn
import torchvision
from models.EDEMA_Net.liftingV4 import LiftingScheme2D

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.disable_conv = in_planes == out_planes
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))

class LevelDAWN(nn.Module):
    def __init__(self, in_planes,kernel_size, share_weights, regu_details, regu_approx):
        super(LevelDAWN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx

        self.wavelet = LiftingScheme2D(in_planes, share_weights,kernel_size=kernel_size)
        self.share_weights = share_weights
        self.bootleneck = BottleneckBlock(4*in_planes , in_planes * 1)

    def forward(self, x):
        (c, d, LL, LH, HL, HH) = self.wavelet(x)
        return self.bootleneck(torch.cat([LL, LH, HL, HH],dim=1))

class Res18WaveletNetBase(nn.Module):
    def __init__(self):
        super(Res18WaveletNetBase, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.conv1.stride = 1
        self.conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.pool0 = LevelDAWN(64,kernel_size=3,
                               share_weights=False,
                               regu_details=0.0, regu_approx=0.0)
        self.l1 = resnet.layer1
        self.pool1 = LevelDAWN(64,kernel_size=3,
                               share_weights=False,
                               regu_details=0.0, regu_approx=0.0)

        self.l2 = resnet.layer2
        self.l2[0].downsample[0].stride = (1,1)
        self.l2[0].conv1.stride = (1,1)
        self.pool2 = LevelDAWN(128,kernel_size=3,
                               share_weights=False,
                               regu_details=0.0, regu_approx=0.0)

        self.l3 = resnet.layer3
        self.l3[0].downsample[0].stride = (1,1)
        self.l3[0].conv1.stride = (1,1)
        self.l4 = resnet.layer4





    def forward(self,x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_features = []
        x = self.conv(x)
        x_features.append(x)

        x = self.pool0(x)
        x = self.l1(x)
        x_features.append(x)

        x = self.pool1(x)
        x = self.l2(x)
        x_features.append(x)

        x = self.pool2(x)
        x = self.l3(x)
        x_features.append(x)
        x = self.l4(x)

        return x,x_features[::-1]




