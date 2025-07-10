# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import logging
import torch.nn as nn
from models.EDEMA_Net.MultiTrans import MultiTrans
from models.EDEMA_Net.WaveletEncoder import Res18WaveletNetBase
logger = logging.getLogger(__name__)
class Down_Conv(nn.Module):
    def __init__(self,size,input_ch,out_ch=128,down=False):
        super(Down_Conv, self).__init__()
        self.down = down
        self.down_s = nn.UpsamplingBilinear2d(size=size)
        self.conv = nn.Sequential(
            nn.Conv2d(input_ch,out_ch,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        if self.down:
            return self.conv(self.down_s(x))
        else:
            return self.conv(x)

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class DecoderCup(nn.Module):
    def __init__(self, decoder_channels = (256, 128, 64, 64),skip_channels = [256, 128, 64, 64],n_skip=4,img_size=(512,256)):
        super().__init__()
        head_channels = 512
        self.img_size = img_size
        self.conv_more = Conv2dReLU(
            512,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.n_skip = n_skip
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        for i in range(4-n_skip):  # re-select the skip channels according to n_skip
            skip_channels[3-i]=0
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

# loss function
def KL(alpha, c=7):
    B,Ch,H,W = alpha.shape
    beta = torch.ones((1, c, H, W)).cuda()
    # beta = torch.ones((1, c, H, W))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def mask_to_onehot(mask, palette=[0,1,2,3,4,5,6]):
    semantic_map = []
    for colour in palette:
        class_map=(mask==colour)
        semantic_map.append(class_map)
    semantic_map = torch.cat(semantic_map, dim=1)
    semantic_map = semantic_map.int()
    return semantic_map

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = mask_to_onehot(p)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = global_step  / annealing_step

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    b = E / (S.expand(E.shape))
    u = c / S

    return b, u, (A + B)


class UnSegNet(nn.Module):
    def __init__(self, img_size=(256,256), num_classes=7, zero_head=False, decoder_channels = (256, 128, 64, 64),
                 lambda_epochs=100):
        super(UnSegNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = "seg"
        self.Baseline = Res18WaveletNetBase()
        self.down_conv = nn.ModuleList()
        channels_features = [256,128,64,64]
        for idx, in_ch in enumerate(channels_features):
            self.down_conv.append(Down_Conv(size=(16,16),input_ch=in_ch,out_ch=512,down=True))


        self.transformer = MultiTrans(dim=512)

        self.decoder = DecoderCup(img_size=img_size)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.final_ac = nn.Softplus()

        self.lambda_epochs = lambda_epochs

        self.classes = num_classes

    def forward(self, x, y, global_step):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        B,c, h, w = x.shape
        x, features = self.Baseline(x) #[2, 512, 32, 16]


        features_norm = []
        for idx in range(len(features)):
            features_norm.append(self.down_conv[idx](features[idx]))



        x = self.transformer(features_norm,x,h,w)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        evidence = self.final_ac(self.segmentation_head(x))


        # step two
        alpha = evidence + 1
        # step three
        b,u,loss = ce_loss(y, alpha, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)

        return b, u, loss


