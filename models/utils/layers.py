import os
import torch
import torch.nn as nn
#ssfrom timm.models.layers import get_padding
from torchvision import models

import torch.nn.functional as F
from torch.nn import init


from models.utils.init_weights import init_weights

#------------------------UNet Layer---------------------
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)  #赋予属性
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)   #获取属性
            x = conv(x)

        return x

class unetConv2_res(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_res, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.conv0 = nn.Conv2d(in_size,out_size,1)
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs_ori = self.conv0(inputs)
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x + inputs_ori

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, is_batchnorm=True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        #print(self.n_concat)
        #print(input)
        outputs0 = self.up(inputs0)
        if input is not None:             
            for i in range(len(input)):
                outputs0 = torch.cat([outputs0,input[i]], 1)
        return self.conv(outputs0)

class unetConv2_SELU(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_SELU, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.SELU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.SELU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp_SELU(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp_SELU, self).__init__()
        self.conv = unetConv2_SELU(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class unetConv2_dilation(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=4, ks=3, stride=1):
        super(unetConv2_dilation, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        s = stride
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, 2**(i-1),2**(i-1)),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p,r),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output = inputs
        #print(output.shape)
        x_0 = inputs
        conv = getattr(self, 'conv1')
        x_1 = conv(x_0)
        conv = getattr(self, 'conv2')
        x_2 = conv(x_1)
        conv = getattr(self, 'conv3')
        x_3 = conv(x_2)
        conv = getattr(self, 'conv4')
        x_4 = conv(x_3)
            

        return x_0 +x_1 +x_2 +x_3 +x_4

class unetConv2_dilation2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=3, ks=3, stride=1):
        super(unetConv2_dilation2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        s = stride
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, 2**(i-1),2**(i-1)),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p,r),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output = inputs
        #print(output.shape)
        x_0 = inputs
        conv = getattr(self, 'conv1')
        x_1 = conv(x_0)
        conv = getattr(self, 'conv2')
        x_2 = conv(x_1)
        conv = getattr(self, 'conv3')
        x_3 = conv(x_2)
            
        return x_0 +x_1 +x_2 +x_3


#--------------------ResNet Layer----------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class resnet_unetUp(nn.Module):
    def __init__(self, in_size, out_size, skip_size, is_deconv):
        super(resnet_unetUp, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, int(in_size/2), kernel_size=4, stride=2, padding=1)
            self.conv = unetConv2(int(in_size/2)+skip_size, out_size, is_batchnorm=True)
        else:
            self.up = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2), 
                    conv1x1(in_size, int(in_size/2)), 
                    nn.BatchNorm2d(int(in_size/2)),
                    nn.ReLU())
            self.conv = unetConv2(int(in_size/2)+skip_size, out_size, is_batchnorm=True)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, inputskip):
        #print(self.n_concat)
        outputs0 = self.up(inputs0)
        
        if inputskip is not None:             
            outputs0 = torch.cat([outputs0,inputskip], 1)
        return self.conv(outputs0)    
  
#--------------------RSBU Layer----------------------
class SBU_Block(nn.Module):
    def __init__(self, channel):
        super(SBU_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   #可以用数学求均值
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, residual):
        agap = self.avg_pool(torch.abs(residual))
        alpha = self.fc(agap)
        soft_threshold = torch.mul(agap, alpha)  #thresold
        residual_out = torch.mul(torch.sign(residual),torch.max((torch.abs(residual)-soft_threshold),torch.tensor(0.0).cuda()))

        return residual_out

class DRSN_Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(DRSN_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('DRSN_Block only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.sbu = SBU_Block(planes)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.sbu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
    
#--------------------SE Layer----------------------
class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel , kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return torch.mul(x,y)

#--------------------SCSE Layer----------------------
class cSE(nn.Module):
    """The channel-wise SE (Squeeze and Excitation) block from the [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) paper.
    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939 and https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Args:
        in_ch (int): The number of channels in the feature map of the input.
        r (int): The reduction ratio of the intermidiate channels.
                Default: 16.
    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch, r=16):
        super(cSE, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x

class sSE(nn.Module):
    """The sSE (Channel Squeeze and Spatial Excitation) block from the
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.
    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Args:
        in_ch (int): The number of channels in the feature map of the input.
    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch):
        super(sSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x

class scSE(nn.Module):
    """The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation) block from the
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Args:
        in_ch (int): The number of channels in the feature map of the input.
        r (int): The reduction ratio of the intermidiate channels.
                Default: 16.
    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch, r=16):
        super(scSE, self).__init__()

        self.SqueezeAndExcitation = cSE(in_ch, r)
        self.ChannelSqueezeAndSpatialExcitation = sSE(
            in_ch
        )

    def forward(self, x):
        cse = self.SqueezeAndExcitation(x)
        sse = self.ChannelSqueezeAndSpatialExcitation(x)

        x = torch.add(cse, sse)

        return x

#--------------------CBAM Layer----------------------
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
        super(ConvBnAct, self).__init__()
        padding = get_padding(kernel_size, stride, dilation)  # assuming PyTorch style padding for this block
        use_aa = aa_layer is not None
        self.conv = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None
        self.drop_block = drop_block
        if act_layer is not None:
            self.act = act_layer(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.act is not None:
            x = self.act(x)
        if self.aa is not None:
            x = self.aa(x)
        return x

class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """
    def __init__(self, channels, reduction=16, act_layer=nn.ReLU):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_avg = self.fc2(self.act(self.fc1(x_avg)))
        x_max = self.fc2(self.act(self.fc1(x_max)))
        x_attn = x_avg + x_max
        return x * x_attn.sigmoid()


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """
    def __init__(self, channels, reduction=16):
        super(LightChannelAttn, self).__init__(channels, reduction)

    def forward(self, x):
        x_pool = 0.5 * self.avg_pool(x) + 0.5 * self.max_pool(x)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * x_attn.sigmoid()


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBnAct(2, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = torch.cat([x_avg, x_max], dim=1)
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """
    def __init__(self, kernel_size=7):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvBnAct(1, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = 0.5 * x_avg + 0.5 * x_max
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class CbamModule(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels)
        self.spatial = SpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x
    
#--------------------=PSP Layer----------------------
class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathnorm=True):
        super().__init__()
        if pool_size == 1 or use_bathnorm == False:                  # PyTorch does not support BatchNorm for 1x1 shape
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),) 
        else:        
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bathnorm=use_bathnorm) for size in sizes
        ])
    
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = self.conv(torch.cat(xs, dim=1))
        return x

class PSPADDModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels, size, use_bathnorm=use_bathnorm) for size in sizes
        ])
    
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = self.conv(xs[0] + xs[1] + xs[2] + xs[3])
        return x
#--------------------=ASPP Layer----------------------     
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,
                dilation=dilation, bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilation,
                dilation=dilation, bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, bias=True,):
        dephtwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=False,)
        pointwise_conv = nn.Conv2d( in_channels, out_channels, kernel_size=1, bias=bias,)
        super().__init__(dephtwise_conv, pointwise_conv)   

#--------------------=CPF Layer----------------------
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn=nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
        self.conv3x3_1=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])
        self.conv3x3_2=nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)



        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x_size= x.size()

        branches_1=self.conv3x3(x)
        branches_1=self.bn[0](branches_1)

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        branches_3=self.bn[2](branches_3)

        feat=torch.cat([branches_1,branches_2],dim=1)
        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat))
        feat=self.relu(self.conv3x3_1[0](feat))
        att=self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)
        
        att_1=att[:,0,:,:].unsqueeze(1)
        att_2=att[:,1,:,:].unsqueeze(1)

        fusion_1_2=att_1*branches_1+att_2*branches_2



        feat1=torch.cat([fusion_1_2,branches_3],dim=1)
        # feat=feat_cat.detach()
        feat1=self.relu(self.conv1x1[0](feat1))
        feat1=self.relu(self.conv3x3_1[0](feat1))
        att1=self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)
        att_3=att1[:,1,:,:].unsqueeze(1)


        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax

class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3]),self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
        feat=self.conv_out(feat)
        return feat
    
class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat=self.conv_out(feat)
        return feat
    
class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
        feat=self.conv_out(feat)
        return feat
      
#--------------------=CE Layer----------------------
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.conv1x1(self.dilate2(x)))
        dilate3_out = self.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = self.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

#--------------------=PSP + SAPF----------------------    
class PSSblock(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([PSPBlock(in_channels, in_channels, size, use_bathnorm=use_bathnorm) for size in sizes])
    
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
    
        self.conv3x3=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=2,dilation=1,kernel_size=3, padding=1)])
    
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self, x):
        branch_1 = self.blocks[0](x)
        branch_2 = self.blocks[1](x)
        branch_3 = self.blocks[2](x)
        branch_4 = self.blocks[3](x)
        
        #branch_1 & branch_2
        feat1 = torch.cat([branch_1,branch_2],dim=1)
        feat1 = self.relu(self.conv1x1[0](feat1))
        att1 = self.relu(self.conv3x3[0](feat1))
        att1 = F.softmax(att1, dim=1)
        
        att1_1 = att1[:,0,:,:].unsqueeze(1)
        att1_2 = att1[:,1,:,:].unsqueeze(1)
        fusion_1 = att1_1*branch_1+att1_2*branch_2
        
        #fusion_1 & branch_3
        feat2 = torch.cat([fusion_1,branch_3],dim=1)
        feat2 = self.relu(self.conv1x1[1](feat2))
        att2 = self.relu(self.conv3x3[1](feat2))
        att2 = F.softmax(att2, dim=1)
        
        att2_1=att2[:,0,:,:].unsqueeze(1)
        att2_2=att2[:,1,:,:].unsqueeze(1)
        fusion_2 = att2_1*fusion_1+att2_2*branch_3
        
        #fusion_2 & branch_4
        feat3 = torch.cat([fusion_2,branch_4],dim=1)
        feat3 = self.relu(self.conv1x1[1](feat3))
        att3 = self.relu(self.conv3x3[1](feat3))
        att3 = F.softmax(att3, dim=1)
        
        att3_1=att2[:,0,:,:].unsqueeze(1)
        att3_2=att2[:,1,:,:].unsqueeze(1)
        fusion_3 = att3_1*fusion_2+att3_2*branch_4
        
        ax = self.relu(self.gamma*fusion_3 + (1-self.gamma)*x)
        out = self.conv_last(ax)
        
        return out