import torch
from torch import nn
from torch.nn import functional as F
import math

class ConvGuidedFilter(nn.Module):
    def __init__(self, nin =3, nout=3, radis=3, eps=0.16):
        super(ConvGuidedFilter, self).__init__()
        self.channal = nout
        padding = int((radis - 1)/2)
        self.eps = eps
        self.box_filter = nn.Conv2d(nin, nout, kernel_size=radis, stride=1, padding=(padding, padding), bias=False, groups=nout)
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr):
        #x_lr为引导图像，y_lr为输入图像
        _, _, h_lrx, w_lrx = x_lr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, self.channal, h_lrx, w_lrx)).fill_(1.0))
        mean_x = self.box_filter(x_lr)/N
        mean_y = self.box_filter(y_lr)/N
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = self.box_filter(A)
        mean_b = self.box_filter(b)
        #mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        #mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * y_lr + mean_b

class ConvGuidedFilterDlicate(nn.Module):
    def __init__(self, nin =3, nout=3, radis=3, eps=0.16, d=1):
        super(ConvGuidedFilterDlicate, self).__init__()
        self.channal = nout
        padding = int((radis - 1)/2) * d
        self.eps = eps
        #self.box_filter = nn.Conv2d(nin, nout, kernel_size=radis, stride=1, padding=(padding, padding), bias=False, groups=nout, dilation=d)
        self.box_filter = nn.Conv2d(nin, nout, kernel_size=radis, stride=1, padding=(padding, padding), bias=False, dilation=d)
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr):
        #x_lr为引导图像，y_lr为输入图像

        _, _, h_lrx, w_lrx = x_lr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, self.channal, h_lrx, w_lrx)).fill_(1.0))
        mean_x = self.box_filter(x_lr)/N
        mean_y = self.box_filter(y_lr)/N
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)

        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = self.box_filter(A)
        mean_b = self.box_filter(b)
        #mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        #mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * y_lr + mean_b

class DDFConvGuidedFilter(nn.Module):
    def __init__(self, nin1=3, nout1=3):
        super(DDFConvGuidedFilter, self).__init__()
        #self.downsample = nn.AvgPool2d(2, stride=2)
        self.ConvGuidedFilter1 = ConvGuidedFilterDlicate(nin =nin1, nout=nin1, radis=3, d=1, eps=0.16)
        self.ConvGuidedFilter2 = ConvGuidedFilterDlicate(nin =nin1, nout=nin1, radis=7, d=1, eps=0.04,)
        self.ConvGuidedFilter3 = ConvGuidedFilterDlicate(nin =nin1, nout=nin1, radis=15, d=1, eps=0.01)
        self.conv1 = nn.Conv2d(3*nin1, nout1, kernel_size=1, stride=1, padding=0, bias=False)
        #self.bnorm = nn.BatchNorm2d(nout1, eps=1e-03)
        #self.act = nn.PReLU(nout1)

    def forward(self ,x_hr):
        _, _, h_hrx, w_hrx = x_hr.size()
        #x_lr = self.downsample(x_hr)
        x_lr = x_hr
        # 多尺度导向滤波
        F1 = self.ConvGuidedFilter1(x_lr, x_lr)
        F2 = self.ConvGuidedFilter2(F1, F1)
        F3 = self.ConvGuidedFilter3(F2, F2)

        #亮细节，小心没有负数
        D1 = x_lr - F1
        D2 = F1 - F2
        D3 = F2 - F3

        #1*1的卷积衡量信息量分配权重
        D_cat = torch.cat([D1, D2, D3], 1)
        D_c = self.conv1(D_cat)
        #D_cat = D1*(self.max_pool(D1)-self.avg_pool(D1)) +D2*(self.max_pool(D2)-self.avg_pool(D2))+ D3*(self.max_pool(D3)-self.avg_pool(D3))

        #升采样
        #out = F.interpolate(D_c, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        out = D_c
        #o_b = self.bnorm(out1)
        #o_a = self.act(o_b)

        return out

class ResDDFConvGuidedFilter(nn.Module):
    def __init__(self, nin1=3):
        super(ResDDFConvGuidedFilter, self).__init__()

        #self.conv1 = nn.Conv2d(nin1, math.ceil(nin1/3), kernel_size=1, stride=1, padding=0, bias=False)
        #self.b1 = nn.BatchNorm2d(math.ceil(nin1/3))
        #self.act1 = nn.PReLU(math.ceil(nin1/3))
        self.DDGF = DDFConvGuidedFilter(nin1=nin1, nout1=nin1)
        self.bnorm = nn.BatchNorm2d(nin1, eps=1e-03)
        self.act2 = nn.PReLU(nin1)

    def forward(self, x_hr):

        #ndown = self.conv1(x_hr)
        #d_bnorm = self.b1(ndown)
        #d_act = self.act1(d_bnorm)
        DDGF = self.DDGF(x_hr)
        asum = DDGF + x_hr
        bnorm = self.bnorm(asum)
        act = self.act2(bnorm)

        return act

