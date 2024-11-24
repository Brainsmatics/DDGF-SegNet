import torch
import torch.nn as nn

__author__ = "Sachin Mehta"

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel):
        super(ChannelAttentionModule, self).__init__()
        ratio = int(channel/2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class Full_AdaptiveReceptiveField(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=False):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 5)  # 预备降到n维（通道数）,等于输出维度/不同膨胀率的卷积核数
        n1 = nOut - 4 * n  # 这n1不就是大于等于五分之一的nout吗
        self.c1 = C(nIn, n, 1, 1)  # 降到n维，即降到输出的五分之一维，在示意图中表示为d
        self.d1 = C(n, n1, 3, 1)
        self.d2 = C(n, n, 5, 1)
        self.d4 = C(n, n, 9, 1)
        self.d8 = C(n, n, 17, 1)
        self.d16 = C(n, n, 33, 1)
        self.cam = ChannelAttentionModule(5)
        self.sigmoid = nn.Sigmoid()

        self.e1 = C(n1, 1, 1, 1)
        self.e2 = C(n, 1, 1, 1)
        self.e4 = C(n, 1, 1, 1)
        self.e8 = C(n, 1, 1, 1)
        self.e16 = C(n, 1, 1, 1)

        #self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)

        # split and transform
        d1 = self.d1(output1)
        d11 = self.e1(d1)

        d2 = self.d2(output1)
        d22 = self.e2(d2)

        d4 = self.d4(output1)
        d44 = self.e4(d4)

        d8 = self.d8(output1)
        d88 = self.e8(d8)

        d16 = self.d16(output1)
        d166 = self.e16(d16)

        # 感受野attention
        combine = torch.cat([d11, d22, d44, d88, d166], 1)
        attent = self.cam(combine)

        da1 = d1 * attent[0, 0, 0, 0]
        da2 = d2 * attent[0, 1, 0, 0]
        da4 = d4 * attent[0, 2, 0, 0]
        da8 = d8 * attent[0, 3, 0, 0]
        da16 = d16 * attent[0, 4, 0, 0]

        # select
        select = max(attent[0, 0, 0, 0], attent[0, 1, 0, 0], attent[0, 2, 0, 0], attent[0, 3, 0, 0], attent[0, 4, 0, 0])

        if attent[0, 0, 0, 0] == select:
            kenel_output = d1
            #c_kenel = self.nn1
        elif attent[0, 1, 0, 0] == select:
            kenel_output = d2
            #c_kenel = self.nn2
        elif attent[0, 2, 0, 0] == select:
            kenel_output = d4
            #c_kenel = self.nn3
        elif attent[0, 3, 0, 0] == select:
            kenel_output = d8
            #c_kenel = self.nn4
        else:
            kenel_output = d16
            #c_kenel = self.nn5

        # merge
        combine1 = torch.cat([da1, da2, da4, da8, da16], 1)

        # if residual version
        if self.add:
            combine1 = input + combine1
        output = combine1
        # output = self.bn(combine)
        lsall=[output, kenel_output]
        return lsall


class AdaptiveReceptiveField(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=False):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 5)  # 预备降到n维（通道数）,等于输出维度/不同膨胀率的卷积核数

        n2 = int(7 * n / 5)
        n3 = int(5 * n / 5)
        n4 = int(3 * n / 5)
        n5 = int(1 * n / 5)
        n1 = nOut - n2 -n3 -n4 -n5
        self.nn1 = n1
        self.nn2 = n2
        self.nn3 = n3
        self.nn4 = n4
        self.nn5 = n5
        #n1 = nOut - 4 * n  # 这n1不就是大于等于五分之一的nout吗

        self.c1 = C(nIn, n, 1, 1)  # 降到n维，即降到输出的五分之一维，在示意图中表示为d
        self.d1 = C(n, n1, 3, 1)
        self.d2 = C(n, n2, 5, 1)
        self.d4 = C(n, n3, 9, 1)
        self.d8 = C(n, n4, 17, 1)
        self.d16 = C(n, n5, 33, 1)
        self.cam = ChannelAttentionModule(5)
        self.sigmoid = nn.Sigmoid()

        self.e1 = C(n1, 1, 1, 1)
        self.e2 = C(n2, 1, 1, 1)
        self.e4 = C(n3, 1, 1, 1)
        self.e8 = C(n4, 1, 1, 1)
        self.e16 = C(n5, 1, 1, 1)

        #self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)

        # split and transform
        d1 = self.d1(output1)
        d11 = self.e1(d1)

        d2 = self.d2(output1)
        d22 = self.e2(d2)

        d4 = self.d4(output1)
        d44 = self.e4(d4)

        d8 = self.d8(output1)
        d88 = self.e8(d8)

        d16 = self.d16(output1)
        d166 = self.e16(d16)

        # 感受野attention
        combine = torch.cat([d11, d22, d44, d88, d166], 1)
        attent = self.cam(combine)

        da1 = d1 * attent[0, 0, 0, 0]
        da2 = d2 * attent[0, 1, 0, 0]
        da4 = d4 * attent[0, 2, 0, 0]
        da8 = d8 * attent[0, 3, 0, 0]
        da16 = d16 * attent[0, 4, 0, 0]


        # merge
        combine1 = torch.cat([da1, da2, da4, da8, da16], 1)

        # if residual version
        if self.add:
            combine1 = input + combine1
        output = combine1
        # output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class U_AdaptiveReceptiveField(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, classes=20):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(1, 25, 3, 1)
        self.level1_ARF = Full_AdaptiveReceptiveField(25, 25)
        self.dsample1 = InputProjectionA(2)

        self.level2 = CBR(25 + 1, 50, 3, 1)
        self.level2_ARF = Full_AdaptiveReceptiveField(50, 50)
        self.b2 = BR(25+1)
        #self.br2 = nn.BatchNorm2d(28, eps=1e-03)
        self.dsample2 = InputProjectionA(1)

        self.level3 = CBR(50 + 25+1, 100, 3, 1)
        self.level3_ARF = AdaptiveReceptiveField(100, 100)
        self.b3 = BR(50 + 25+1)
        #self.br3 = nn.BatchNorm2d(50+28, eps=1e-03)
        self.dsample3 = InputProjectionA(1)

        self.level4 = CBR(176, 200, 3, 1)
        self.level4_2 = CBR(200, 200, 3, 1)
        self.b4 = BR(176)

        #解码
        self.upsample4 = nn.ConvTranspose2d(200, 200, 2, stride=2, padding=0, output_padding=0, bias=False)

        self.b33 = BR(100+200)
        self.level3_o = CBR(100+200, 100, 3, 1)
        self.level3_o2 = CBR(100, 100, 3, 1)
        self.upsample3 = nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0, output_padding=0, bias=False)

        self.b22 = BR(10+100)
        self.level2_o = CBR(10+100, 50, 3, 1)
        self.level2_o2 = CBR(50, 50, 3, 1)
        self.upsample2 = nn.ConvTranspose2d(50, 50, 4, stride=4, padding=0, output_padding=0, bias=False)

        self.b11 = BR(5+50)
        self.level1_o = CBR(5 + 50, 25, 3, 1)
        self.classifier = C(25, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        #编码器
        op1 = self.level1(input)
        a1 = self.level1_ARF(op1)
        ar1, arc1 = a1[0], a1[1]

        input2 = self.dsample1(ar1)
        input2_o = self.dsample1(input)
        concat2 = torch.cat([input2_o, input2], 1)
        concat2b = self.b2(concat2)
        op2 = self.level2(concat2b)
        a2 = self.level2_ARF(op2)
        ar2, arc2 = a2[0], a2[1]


        input3 = self.dsample2(ar2)
        input3_o = self.dsample2(concat2b)
        concat3 = torch.cat([input3_o, input3], 1)
        concat3b = self.b3(concat3)
        op3 = self.level3(concat3b)
        ar3 = self.level3_ARF(op3)


        input4 = self.dsample3(ar3)
        input4_o = self.dsample3(concat3)
        concat4 = torch.cat([input4_o, input4], 1)
        concat4b = self.b4(concat4)
        op4 = self.level4(concat4b)
        op4_2 = self.level4_2(op4)

        #解码器

        up3 = self.b33(torch.cat([ar3 ,self.upsample4(op4_2)], 1))
        up3_o = self.level3_o(up3)
        up3_o2 = self.level3_o2(up3_o)

        up2 = self.b22(torch.cat([arc2, self.upsample3(up3_o2)], 1))
        up2_o = self.level2_o(up2)
        up2_o2 = self.level2_o2(up2_o)

        up1 = self.b11(torch.cat([arc1, self.upsample2(up2_o2)], 1))
        up1_o = self.level1_o(up1)
        classifier = self.classifier(up1_o)

        return classifier