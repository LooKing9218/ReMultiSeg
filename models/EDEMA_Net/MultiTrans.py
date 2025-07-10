import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim,size=(16,16)):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.rel_h = nn.Parameter(torch.randn([1, in_dim, size[0], 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, in_dim, 1, size[1]]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy_Content = torch.bmm(proj_query, proj_key)

        content_position = (self.rel_h + self.rel_w).view(1, -1, width*height)
        content_position = torch.matmul(proj_query,content_position)

        energy = energy_Content + content_position
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return out



class MultiTrans(nn.Module):
    def __init__(self,dim=512,num_heads=5):
        super(MultiTrans, self).__init__()
        self.num_heads = num_heads
        self.SA_Heads = nn.ModuleList()
        for idx in range(num_heads):
            self.SA_Heads.append(Self_Attention(in_dim=dim))
        self.fusion_multi_scale = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.multi_scale_weights = nn.Sequential(
            nn.Conv2d(num_heads*dim,num_heads-1,kernel_size=1,bias=False),
            nn.BatchNorm2d(num_heads-1),
            nn.Sigmoid()
        )

        self.SA_Top_Ms_Heads = nn.ModuleList()
        for idx in range(2):
            self.SA_Top_Ms_Heads.append(Self_Attention(in_dim=dim))
        self.fusion_Top_Ms = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.TopMs_BNRelu = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )


    def forward(self,x_list,x,h,w):
        h_x = x
        x_list_Temp = x_list.copy()
        x_list_Temp.append(x)
        x_mult_scale = []
        for idx in range(self.num_heads):
            x_ms = self.SA_Heads[idx](x_list_Temp[idx])
            x_mult_scale.append(x_ms)
        weights = self.multi_scale_weights(torch.cat(x_mult_scale,dim=1))


        x_Ms = torch.unsqueeze(weights[:,0,:,:],dim=1)*x_mult_scale[0]+ \
               torch.unsqueeze(weights[:, 1, :, :], dim=1) * x_mult_scale[1] +\
               torch.unsqueeze(weights[:, 2, :, :], dim=1) * x_mult_scale[2] +\
               torch.unsqueeze(weights[:, 3, :, :], dim=1) * x_mult_scale[3] + x_mult_scale[4]

        x_Ms = self.fusion_multi_scale(x_Ms)
        x_Ms = self.TopMs_BNRelu(x_Ms+h_x)
        return x_Ms




