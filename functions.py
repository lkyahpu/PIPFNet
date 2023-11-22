import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import threading
# import globals

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:  -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        # n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        # hr_x = hr_x.double()
        l_a = l_a.double()

        # assert n_lrx == n_lry and n_lry == n_hrx
        # assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        # assert h_lrx == h_lry and w_lrx == w_lry
        # assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        # mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        # mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (A * lr_x + b).float()

class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        # theta_x_size = theta_x.size()

        phi_g = self.phi(g)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))

        return sigm_psi_f




def eta_l1(r_, lam_):
    # l_1 norm based
    # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
    B, C, H, W = r_.shape
    lam_ = torch.reshape(lam_, [1, C, 1, 1])
    lam_ = lam_.repeat(B, 1, H, W)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R


class LRR_Block_lista(nn.Module):
    def __init__(self, s, n, c, stride):
        super(LRR_Block_lista, self).__init__()
        self.conv_Wdz = ConvLayer(n, c, s, stride)
        self.conv_Wdtz = ConvLayer(c, n, s, stride)

    def forward(self, x, tensor_z, lam_theta, lam_z):
        # Updating
        convZ1 = self.conv_Wdz(tensor_z)
        midZ = x - convZ1
        tensor_c = lam_z * tensor_z + self.conv_Wdtz(midZ)
        # tensor_c = tensor_b + hZ
        Z = eta_l1(tensor_c, lam_theta)
        return Z


class GetLS_Net(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Net, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3


        self.n = n
        self.num_block = num_block
        self.conv_W00 = ConvLayer(channel, 2 * n, s, stride)
        self.lamj = nn.Parameter(torch.rand(1, self.n * 2))  # l1-norm
        self.lamz = nn.Parameter(torch.rand(1, 1))
        self.up = nn.Upsample(scale_factor=2)
        for i in range(num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))

    def forward(self, x):


        b, c, h, w = x.shape
        tensor_l = self.conv_W00(x)  # Z_0
        tensor_z = eta_l1(tensor_l, self.lamj)

        for i in range(self.num_block):
            # print('num_block - ' + str(i))
            lrrblock = getattr(self, 'lrrblock' + str(i))
            tensor_z = lrrblock(x, tensor_z, self.lamj, self.lamz)
        L = tensor_z[:, :self.n, :, :]
        S = tensor_z[:, self.n: 2 * self.n, :, :]


        # globals.nm+=1
        #
        # if globals.nm==4:
        #     visual_feature(L, './visual/b_lr/dolp/L')
        #     visual_feature(S, './visual/b_lr/dolp/S')

        return L, S






class HOR(nn.Module):
    def __init__(self,channel):
        super(HOR, self).__init__()
        self.high = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.low = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

        self.value = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.e_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.mid = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.latter = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

    def forward(self, x_latter, x):
        b, c, h, w = x_latter.shape
        _, c_, _, _ = x.shape
        x_latter_ = self.high(x_latter).reshape(b, c, h * w).contiguous()
        x_ = self.low(x).reshape(b, c_, h * w).permute(0, 2, 1).contiguous()

        p = torch.bmm(x_latter_, x_).contiguous()
        p = self.softmax(p).contiguous()

        e_ = torch.bmm(self.value(x).reshape(b, c, h * w).permute(0, 2, 1), p).contiguous()
        e = e_ + x_
        e = e.permute(0, 2, 1).contiguous()
        e = self.e_conv(e.reshape(b, c, h, w)).reshape(b, c, h * w).contiguous()

        # e = e.permute(0, 2, 1)
        x_latter_ = self.latter(x_latter).reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        t = torch.bmm(e, x_latter_).contiguous()
        t = self.softmax(t).contiguous()

        x_ = self.mid(x).view(b, c_, h * w).permute(0, 2, 1).contiguous()
        out = torch.bmm(x_, t).permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        return out

class PSCA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA)
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Conv2d(d_model, d_ff, 1, bias=False)
        self.w_2 = nn.Conv2d(d_ff, d_model, 1, bias=False)
        self.w_3 = nn.Conv2d(d_model, d_model, 1, bias=False)

        nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        return x

class FA(nn.Module):

    def __init__(self, in_dim):

        super(FA, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_s0 = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_dolp = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, s0, dolp):

        attmap = self.conv( torch.cat( (s0,dolp),1) )
        attmap = torch.sigmoid(attmap)

        s0_f = attmap[:,0:1,:,:] * s0 * self.v_s0
        dolp_f = attmap[:,1:,:,:] * dolp * self.v_dolp

        # s0_f = attmap[:, 0:1, :, :] * s0
        # s0_f = s0_f * self.v_s0
        #
        # dolp_f = attmap[:, 1:, :, :] * dolp
        # dolp_f =  dolp_f *self.v_dolp

        # visual_single(attmap[:, 1:, :, :], './visual/b_lr_agf/S0', 's0')
        # visual_single(attmap[:, 0:1, :, :], './visual/b_lr_agf/dolp', 'dolp')

        out = s0_f + dolp_f

        return out



class Att_map(nn.Module):
    """ Attentive Map
    """

    def __init__(self, channel):
        super().__init__()
        # self.weight = nn.Sequential(
        #     nn.Conv3d(channel * 2, channel, 1),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(channel, channel, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        self.get_ls = GetLS_Net(s=1, n= channel, channel= channel, stride=1, num_block=2)


        self.theta = nn.Conv2d(in_channels= 2*channel, out_channels=channel, kernel_size=1)

        self.psi = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0,
                             bias=True)

    def forward(self, x, y):
        # w = F.relu(self.theta(torch.cat([x, y], dim=1)))
        # L, S = self.get_ls(w)
        # out = torch.sigmoid(L) * x + torch.sigmoid(S) * y
        # out = self.psi(out)

        w = F.relu(self.theta(torch.cat([x, y], dim=1)))
        L, S = self.get_ls(w)
        out = L * x + S * y
        out = torch.sigmoid(self.psi(out))

        return out

class ASC(nn.Module):
    """ Attentive Skip Connection
    """

    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

        # self.channel_1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        # self.s0 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        #
        # self.dolp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)



    def forward(self, x, y):
        # x_s0 = self.s0(x)
        #
        # x_dolp = self.dolp(x)

        w = self.weight(torch.cat([x, y], dim=1))

        out = (1 - w) * x + w * y

        # out = self.channel_1(out)
        #
        #
        # out = torch.sigmoid(out)

        return out


def batch_pix_accuracy(output, target):

    # if len(target.shape) == 3:
    #     target = np.expand_dims(target.float(), axis=1)
    # elif len(target.shape) == 4:
    #     target = target.float()
    # else:
    #     raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0)
    pixel_labeled = (target > 0).sum()
    pixel_correct = (((predict == target))*((target > 0))).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0)
    # if len(target.shape) == 3:
    #     target = np.expand_dims(target.float(), axis=1)
    # elif len(target.shape) == 4:
    #     target = target.float()
    # else:
    #     raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target))

    area_inter, _  = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    # assert (area_inter <= area_union).all(), \
    #     "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


def IoU(preds, labels):

    correct, labeled = batch_pix_accuracy(preds, labels)
    inter, union = batch_intersection_union(preds, labels)

    pixAcc = 1.0 * correct / (np.spacing(1) + labeled)
    IoU = 1.0 * inter / (np.spacing(1) + union)
    # mIoU = IoU.mean()

    return pixAcc, IoU


def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return prec, recall, F1



class ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (F.sigmoid(output).detach().numpy() > score_thresh).astype('int64') # P
    target = target.detach().numpy().astype('int64')  # T
    intersection = predict * (predict == target) # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    return tp, pos, fp, neg


def visual_feature(x,dir_path):

    x = x.squeeze()
    x = x.detach().cpu().numpy()
    for ii in range(x.shape[0]):

        x_batch = x[ii,:,:].squeeze()

        # x_batch = (x_batch - np.min(x_batch)) / (np.max(x_batch) - np.min(x_batch)+np.spacing(1))
        #
        # x_batch = (x_batch*255)#.astype('uint8')

        plt.figure()
        ax = plt.gca()
        im = ax.imshow(x_batch, cmap='bwr')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.axis('off')
        plt.savefig(dir_path + "/heat_{}.png".format(ii), bbox_inches='tight', pad_inches=0.0, format="png", dpi=300)

        # heat_color = cv2.applyColorMap(x_batch,2)
        #
        # cv2.imwrite(dir_path + "/heat_{}.png".format(ii), heat_color)

def visual_single(x,dir_path,name):

    x = x.squeeze()
    x_batch = x.detach().cpu().numpy()

    # x_batch = x[ii,:,:].squeeze()

    x_batch = (x_batch - np.min(x_batch)) / (np.max(x_batch) - np.min(x_batch))
    #
    x_batch = (x_batch*255).astype('uint8')

    # heat_color = cv2.applyColorMap(x_batch,0)

    # fig=plt.figure(name)
    # ax = plt.axes()
    # cax = fig.add_axes([ax.get_position().x1 + 0.015, ax.get_position().y0, 0.02, ax.get_position().height])
    # plt.colorbar(x_batch, cax=cax)
    # plt.imshow(x_batch, cmap='bwr')
    # plt.axis('off')
    # plt.savefig(dir_path + "/heat.svg",  format="svg")  #bbox_inches='tight', pad_inches=0.0,

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(x_batch, cmap='bwr')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis('off')
    plt.savefig(dir_path + "/heat.svg",  bbox_inches='tight', pad_inches=0.0, format="svg")


    # cv2.imwrite(dir_path + "/heat.png", heat_color)













