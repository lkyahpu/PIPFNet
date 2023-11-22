# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functions import AdaIN, FastGuidedFilter_attention, GridAttentionBlock, GetLS_Net, FA, Att_map, ASC
import collections




class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.conv_last = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5
        # return torch.tanh(self.conv(x))


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True, type='S0'):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        # self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)  # surrounding context
        self.get_ls = GetLS_Net(s=1, n=n, channel=n, stride=1, num_block=6)
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)
        # self.sobelconv = Sobelxy(nIn)
        self.convup = Conv1(nIn, nOut)
        self.nout = nOut
        # self.h_size = 256
        # self.w_size = 320
        self.type = type

        # self.FCA = MultiSpectralAttentionLayer(self.nout, self.h_size, self.w_size)

        # self.FCA = MultiSpectralAttentionLayer(nOut, h_size, w_size)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        # sur = self.F_sur(output)



        if self.type == 'S0':

            input_ls, _ = self.get_ls(output)

        else:
            _, input_ls = self.get_ls(output)

        joi_feat = torch.cat([loc, input_ls], 1)


        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature




        # output = self.FCA(joi_feat)

        # output = self.FCA(joi_feat)

        # if residual version
        if self.add:
            # input_res=self.sobelconv(input)
            input_res = self.convup(input)
            output = input_res + output
        return output


class CNN3(nn.Module):
    def __init__(self, channels=[3 * 42 * 42, 3 * 42 * 7, 3 * 7 * 7, 3 * 7, 3]):
        super(CNN3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(channels[0], channels[1], 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(channels[1], channels[2], 1, 1, 0, bias=True)
        self.e_conv3 = nn.Conv2d(channels[2], channels[3], 1, 1, 0, bias=True)
        self.e_conv4 = nn.Conv2d(channels[3], channels[4], 1, 1, 0, bias=True)

    def forward(self, x):
        x1 = self.relu((self.e_conv1(x)))  # 8
        x2 = self.relu((self.e_conv2(x1)))  # 4
        x3 = self.relu((self.e_conv3(x2)))  # 2
        y = self.e_conv4(x3)  # 1
        return y


class FusionNet(nn.Module):
    def __init__(self, output=1, type='train'):
        super(FusionNet, self).__init__()
        vis_ch = [16, 16, 32]
        inf_ch = [16, 16, 32]
        output = 1

        self.s0_branch = nn.Sequential(collections.OrderedDict([
            ('conv_s0', ConvLeakyRelu2d(1, vis_ch[0])),
            ('CG1_s0', ContextGuidedBlock(vis_ch[0], vis_ch[1],  type='S0')),
            ('CG2_s0', ContextGuidedBlock(vis_ch[1], vis_ch[2],  type='S0')),

        ]))

        self.dolp_branch = nn.Sequential(collections.OrderedDict([
            ('conv_dolp', ConvLeakyRelu2d(1, vis_ch[0])),
            ('CG1_dolp', ContextGuidedBlock(vis_ch[0], vis_ch[1],  type='dolp')),
            ('CG2_dolp', ContextGuidedBlock(vis_ch[1], vis_ch[2],  type='dolp')),

        ]))

        self.fusion_decoder = nn.Sequential(collections.OrderedDict([
            ('decode3', ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1])),
            ('decode2', ConvBnLeakyRelu2d(vis_ch[1], vis_ch[0])),
            ('decode1', ConvBnLeakyRelu2d(vis_ch[0], vis_ch[0])),

        ]))

        # self.seg_out = nn.Sequential(collections.OrderedDict([
        #     ('seg_conv', nn.Conv2d(vis_ch[0], vis_ch[0], 1)),
        #     ('seg_Bn', nn.BatchNorm2d(vis_ch[0])),
        #     ('seg_Relu', nn.ReLU(inplace=True)),
        #     ('seg_out', nn.Conv2d(vis_ch[0], 1, 1)),
        #
        # ]))

        self.fusion_out = nn.Sequential(collections.OrderedDict([
            ('fusion_out', ConvBnTanh2d(vis_ch[0], output)),
        ]))

        # self.conv2to2 = nn.Sequential(collections.OrderedDict([
        #     ('conv2to1', ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[2]+inf_ch[2])),
        # ]))
        #

        # self.conv2to1 = nn.Sequential(collections.OrderedDict([
        #     ('conv2to1', ConvBnLeakyRelu2d(vis_ch[2] + inf_ch[2], vis_ch[2])),
        # ]))

        # self.adain = AdaIN()

        self.gf = FastGuidedFilter_attention(r=11, eps=1e-5)

        self.attmap = GridAttentionBlock(vis_ch[2])

        # self.attmap = Att_map(vis_ch[2])
        # self.attmap = ASC(vis_ch[2])

        self.FA = FA(vis_ch[2])

        self.h_size = 256
        self.w_size = 320

        # self.FCA = MultiSpectralAttentionLayer(vis_ch[2]+ inf_ch[2], self.h_size, self.w_size)

    def forward(self, image_s0, image_dolp):
        x_s0 = self.s0_branch(image_s0)
        x_dolp = self.dolp_branch(image_dolp)


        attmap = self.attmap(x_s0, x_dolp)
        fusion = self.gf(x_s0, x_dolp, attmap)

        # fusion = torch.cat((fusion, x_s0), dim=1)
        fusion = self.FA(fusion, x_s0)


        x_fusion = self.fusion_decoder(fusion)
        x = self.fusion_out(x_fusion)
        # x_seg = self.seg_out(x_fusion)

        x_en_s0 = self.s0_branch(x)
        x_en_dolp = self.dolp_branch(x)

        x_de_s0 = self.fusion_decoder(x_en_s0)
        x_de_s0 = self.fusion_out(x_de_s0)

        x_de_dolp = self.fusion_decoder(x_en_dolp)
        x_de_dolp = self.fusion_out(x_de_dolp)

        return x, x_de_s0, x_de_dolp, x_s0, x_dolp, x_en_s0, x_en_dolp



def unit_test():
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    x1 = torch.tensor(np.random.rand(2, 1, 256, 320).astype(np.float32))
    x2 = torch.tensor(np.random.rand(2, 1, 256, 320).astype(np.float32))
    model = FusionNet(output=1)
    model = model.to(device)

    # model_seg = joint_model()
    # model_seg = model_seg.to(device)

    x1 = x1.to(device)
    x2 = x2.to(device)
    # denoise = denoiseNet_adaptive(in_size=448)
    # z = denoise(x1)
    # print(z.shape)
    y, _, _, _, _, _, _ = model(x1, x2)
    # y, _, _, _, _, _, _, _ = model_seg(x1, x2)

    print('output shape:', y.shape)
    assert y.shape == (2, 1, 256, 320), 'output shape (2,1,480,640) is expected!'
    print('test ok!')
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.3fM" % (total / 1e6))

    # mat1 = torch.randn(4, 32, 320, 256)
    # mat2 = torch.randn(4, 1, 320, 256)
    # out = torch.mul(mat1, mat2)
    # print(out.shape)
    # torch.Size([2, 1, 4, 2])


if __name__ == '__main__':
    unit_test()
