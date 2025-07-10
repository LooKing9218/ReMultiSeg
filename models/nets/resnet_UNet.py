import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from models.utils.layers import conv1x1,BasicBlock,Bottleneck,resnet_unetUp
from models.utils.init_weights import init_weights
#from .utils import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class resnet_UNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, num_classes=3, is_deconv =True,
                 zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None):
        super(resnet_UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        #downsampling
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #upsampling
        #Deconv上采样时通道数减半，in_channels=[512,256,128,96,16]
        #upsample上采样时通道数不变，in_channels=[512+256,256+128,128+64,96+64,32+0]
        #self.up_concat3 = resnet_unetUp(in_size=512, out_size=256, skip_size=256, is_deconv=self.is_deconv)
        self.up_concat3 = resnet_unetUp(in_size=2048, out_size=1024, skip_size=1024, is_deconv=self.is_deconv)
        # self.up_concat2 = resnet_unetUp(in_size=256, out_size=128, skip_size=128, is_deconv=self.is_deconv)
        self.up_concat2 = resnet_unetUp(in_size=1024, out_size=512, skip_size=512, is_deconv=self.is_deconv)
        #self.up_concat1 = resnet_unetUp(in_size=128, out_size=64, skip_size=64, is_deconv=self.is_deconv)
        self.up_concat1 = resnet_unetUp(in_size=512, out_size=256, skip_size=256, is_deconv=self.is_deconv)
        #self.up_concatcbr = resnet_unetUp(in_size=64, out_size=32, skip_size=64, is_deconv=self.is_deconv)
        self.up_concatcbr = resnet_unetUp(in_size=256, out_size=128, skip_size=64, is_deconv=self.is_deconv)
        #self.up_sample = resnet_unetUp(in_size=32, out_size=16, skip_size=0, is_deconv=self.is_deconv)  # 没有skip，仅上采样至原图大小
        self.up_sample = resnet_unetUp(in_size=128, out_size=64, skip_size=0, is_deconv=self.is_deconv) #没有skip，仅上采样至原图大小
        #segmentation_head
        #self.final = nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)

        # initialise weights 这里mode='fan_in'or'fan_out'可以改
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
        
    def _forward_impl(self, inputs):                 # 1*1024*512 原图
        # See note [TorchScript super()]
        conv1 = self.conv1(inputs)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)                    # 64*512*256（步长为2的7*7卷积）
        
        conv2_x = self.maxpool(conv1)
        conv2_x = self.layer1(conv2_x)              # 64*256*128
        
        conv3_x = self.layer2(conv2_x)              # 128*128*64
        
        conv4_x = self.layer3(conv3_x)              # 256*64*32
        
        conv5_x = self.layer4(conv4_x)              # 512*32*16（下采样8倍）  
        
        up4 = self.up_concat3(conv5_x,conv4_x)      # 256*64*32
        up3 = self.up_concat2(up4,conv3_x)          # 128*128*64
        up2 = self.up_concat1(up3,conv2_x)          # 64*256*128
        up1 = self.up_concatcbr(up2,conv1)          # 32*512*256
        up0 = self.up_sample(up1,None)              # 16*1024*512（没有skip,仅上采样）
        
        final = self.final(up0)                     # 4*1024*512
   
        return F.log_softmax(final,dim=1)

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = resnet_UNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34_UNet(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_UNet(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


#------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = resnet50_UNet(in_channels=3, num_classes=3, is_deconv=True, pretrained=False).cuda()
    x = torch.rand((4, 3, 256, 128)).cuda()
    forward = net.forward(x)
#    print(forward)
#    print(net)
    torchsummary.summary(net, (3, 256, 128))
