import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layers import unetConv2,unetUp,unetConv2_dilation
from models.utils.init_weights import init_weights

class UNet(nn.Module):

    def __init__(self, in_channels=1,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
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

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
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
        if inputs.size()[1] == 1:
            inputs = inputs.repeat(1,3,1,1)
        conv1 = self.conv1(inputs)       # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)     # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)     # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)     # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)   # 256*32*64

        up4 = self.up_concat4(center,conv4)  # 128*64*128
        up3 = self.up_concat3(up4,conv3)     # 64*128*256
        up2 = self.up_concat2(up3,conv2)     # 32*256*512
        up1 = self.up_concat1(up2,conv1)     # 16*512*1024

        final_1 = self.final_1(up1)

        return F.log_softmax(final_1,dim=1)


class UNet_m(nn.Module):

    def __init__(self, in_channels=6,n_classes=3,feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_m, self).__init__()
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

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
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
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        final_1 = self.final_1(up1)

        return F.log_softmax(final_1, dim=1)


class UNet_multi(nn.Module):

    def __init__(self, in_channels=1,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_multi, self).__init__()
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

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256,3,1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
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
        conv1 = self.conv1(inputs)       # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)     # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)     # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)     # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)   # 256*32*64
        cls_branch = self.cls(center).squeeze()

        up4 = self.up_concat4(center,conv4)  # 128*64*128
        up3 = self.up_concat3(up4,conv3)     # 64*128*256
        up2 = self.up_concat2(up3,conv2)     # 32*256*512
        up1 = self.up_concat1(up2,conv1)     # 16*512*1024

        final_1 = self.final_1(up1)
        # final_2 = self.final_2(up1)
        # final_3 = self.final_3(up1)

        return F.log_softmax(final_1,dim=1),cls_branch

class FC(nn.Module):

    def __init__(self, in_channels=512,n_classes=5,is_batchnorm=True):
        super(FC, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # downsampling
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, 2, 1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 2, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True), )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 2, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True), )
        self.fc1 = nn.Linear(1024,128)    #center=fc1(center.view(4,-1))
        self.fc2 = nn.Linear(128,5)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)   # 256*16*16
        conv2 = self.conv2(conv1)    # 128*8*8
        conv3 = self.conv3(conv2)    # 64*4*4

        fc1 = self.fc1(conv3.view(8,-1))        # 128
        fc2 = self.fc2(fc1)        # 5

        return F.softmax(fc2,dim=1)

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=1,n_classes=7,feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds=True):
        super(UNet_Nested, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256,3,1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv,3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv,3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv,3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv,4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv,4)
        
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv,5)
        
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       
        maxpool0 = self.maxpool0(X_00)  
        X_10= self.conv10(maxpool0)     
        maxpool1 = self.maxpool1(X_10) 
        X_20 = self.conv20(maxpool1)    
        maxpool2 = self.maxpool2(X_20)  
        X_30 = self.conv30(maxpool2)     
        maxpool3 = self.maxpool3(X_30)  
        X_40 = self.conv40(maxpool3)   
        cls_branch = self.cls(X_40).squeeze()
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1+final_2+final_3+final_4)/4


        return F.log_softmax(final,dim=1),cls_branch



class UNet_Nested_dilated(nn.Module):

    def __init__(self, in_channels=3,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds=True):
        super(UNet_Nested_dilated, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.dilated = unetConv2_dilation(filters[4],filters[4],self.is_batchnorm)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256,3,1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv,3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv,3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv,3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv,4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv,4)
        
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv,5)
        
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)      
        maxpool0 = self.maxpool0(X_00)  
        X_10= self.conv10(maxpool0)     
        maxpool1 = self.maxpool1(X_10)  
        X_20 = self.conv20(maxpool1)     
        maxpool2 = self.maxpool2(X_20)  
        X_30 = self.conv30(maxpool2)     
        maxpool3 = self.maxpool3(X_30)  
        X_40 = self.conv40(maxpool3)   
        X_40_d = self.dilated(X_40)
        cls_branch = self.cls(X_40_d).squeeze()
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40_d,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1+final_2+final_3+final_4)/4

        if self.is_ds:
            return F.log_softmax(final),cls_branch
        else:
            return F.log_softmax(final_4),cls_branch

#------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = UNet(in_channels=1, n_classes=4, is_deconv=True).cuda()
    print(net)
    x = torch.rand((4, 1, 256, 128)).cuda()
    forward = net.forward(x)
    print(forward)
    print(type(forward))
    
#    net = resnet34_unet(in_channel=1,out_channel=4,pretrain=False).cuda()
#    torchsummary.summary(net, (1, 512, 512))