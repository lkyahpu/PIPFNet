#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import models
import numpy as np
import math


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class NormalLoss(nn.Module):
    def __init__(self, ignore_lb=255, *args, **kwargs):
        super(NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.TV = TVLoss()

    def forward(self, image_S0, image_dolp, generate_img):  # labels,,i
        image_y = image_S0  # [:,:1,:,:]
        x_in_max = torch.max(image_y, image_dolp)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_dolp)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        # loss_total=loss_in+loss_grad #+self.TV(x1-x2)

        return loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map, sigma1_sq


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    value, sigma1_sq = _ssim(img1, img2, window, window_size, channel, size_average)
    v = torch.zeros_like(sigma1_sq) + 0.0001
    sigma1 = torch.where(sigma1_sq < 0.0001, v, sigma1_sq)
    return value, sigma1


def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])


def sigma(x):
    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
    across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
    the permutations are required for broadcasting"""
    return torch.sqrt(
        torch.sum((x.permute([2, 3, 0, 1]) - mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) / (x.shape[2] * x.shape[3]))


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


class RegularizedLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()

        self.gamma = gamma
        # self.L_Grad1 = L_Grad()
        # self.L_Inten = L_Intensity()
        self.mae = nn.L1Loss()
        # self.percept_loss = VGG_percept_loss()
        self.TVLoss = TVLoss()
        self.sobelconv = Sobelxy()

        self.IGLoss = Fusionloss()
        self.contrast_s0 = ContrastLoss(ablation=True)
        self.contrast_dolp = ContrastLoss()
        self.contrast = ContrastLoss(ablation=True)
        # self.L_Grad_single = L_Grad_single()

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map, sigma1_sq

    def ssim(self, img1, img2, window_size=11, size_average=True):
        img1 = torch.clamp(img1, min=0, max=1)
        img2 = torch.clamp(img2, min=0, max=1)
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        value, sigma1_sq = self._ssim(img1, img2, window, window_size, channel, size_average)
        v = torch.zeros_like(sigma1_sq) + 0.0001
        sigma1 = torch.where(sigma1_sq < 0.0001, v, sigma1_sq)
        return value, sigma1

    def mseloss(self, image, target):
        x = ((image - target) ** 2)
        return torch.mean(x)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1 - g2 - G1 + G2) ** 2)

    def mssim_loss(self, img1, img2, img):
        # img1, img2 = tf.split(y_, 2, 3)
        img3 = img1 * 0.5 + img2 * 0.5
        Win = [11, 9, 7, 5, 3]
        loss = 0

        for s in Win:
            loss1, sigma1 = self.ssim(img1, img, s)
            loss2, sigma2 = self.ssim(img2, img, s)

            r = sigma1 / (sigma1 + sigma2 + 0.0000001)
            tmp = 1 - torch.mean(r * loss1) - torch.mean((1 - r) * loss2)
            # tmp = 1 - w1*torch.mean(loss1) - w2*torch.mean(loss2)

            loss = loss + tmp
        loss = loss / 5.0
        # loss = loss + torch.mean(torch.abs(img3 - img)) * 0.1
        return loss

    def mssim3_loss(self, img1, img2, img3, img):
        # img1, img2 = tf.split(y_, 2, 3)
        img4 = (img1 + img2 + img3) / 3.0
        Win = [11, 9, 7, 5, 3]
        loss = 0

        for s in Win:
            loss1, sigma1 = self.ssim(img1, img, s)
            loss2, sigma2 = self.ssim(img2, img, s)
            loss3, sigma3 = self.ssim(img3, img, s)

            r1 = sigma1 / (sigma1 + sigma2 + sigma3 + 0.0000001)
            r2 = sigma2 / (sigma1 + sigma2 + sigma3 + 0.0000001)
            r3 = sigma3 / (sigma1 + sigma2 + sigma3 + 0.0000001)

            tmp = 1 - torch.mean(r1 * loss1) - torch.mean(r2 * loss2) - torch.mean(r3 * loss3)
            # tmp = 1 - w1*torch.mean(loss1) - w2*torch.mean(loss2)

            loss = loss + tmp
        loss = loss / 5.0
        # loss = loss + torch.mean(torch.abs(img4 - img)) * 0.1
        return loss

    def grad(self, image_A, image_B, image_C, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        gradient_joint = torch.max(gradient_joint, gradient_C)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

    def styleLoss(self, style, output):
        mu_sum = torch.norm(mu(style) - mu(output))
        sigma_sum = torch.norm(sigma(style) - sigma(output))
        return mu_sum + sigma_sum

    def polar_loss(self, s0, dolp, aop, img_s0):

        I_0 = 0.5 * s0 * (1 - torch.mul(dolp, torch.cos(2 * (aop * math.pi - 0))))
        I_60 = 0.5 * s0 * (1 - torch.mul(dolp, torch.cos(2 * (aop * math.pi - math.pi / 3.0))))
        I_120 = 0.5 * s0 * (1 - torch.mul(dolp, torch.cos(2 * (aop * math.pi - 2.0 * math.pi / 3.0))))

        I_s0 = 2*(I_0 + I_60 + I_120) / 3.0

        Q_s1 = ((2 * I_0 - I_60 - I_120) * 2.0) / 3.0

        U_s2 = ((I_60 - I_120) * 2.0) / math.sqrt(3.0)

        P_dolp = torch.sqrt(torch.pow(Q_s1, 2) + torch.pow(U_s2, 2) +1e-7) / (I_s0+1e-7)

        # P_dolp = torch.pow(Q_s1, 2) + torch.pow(U_s2, 2) / (torch.pow(I_s0 , 2) + 1e-8)

        s0_loss = self.mae(I_s0, img_s0)
        # dolp_loss = self.mae(torch.pow(dolp, 2), P_dolp)

        # s1_s0_loss = self.mae(s1_s0, Q_s1/(I_s0 + 1e-8))
        dolp_loss = self.mae(dolp, P_dolp)
        # print(s0_loss,dolp_loss)

        return s0_loss #+dolp_loss

    def seg_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def forward(self, img_ir, img_p, output_pf, s0_de, dolp_de, s1_s0, aop, x_s0, x_dolp, x_en_s0, x_en_dolp):  #, x_s0, x_dolp, x_s0_en, x_dolp_en, s1_s0_en

        # p_loss = self.mssim_loss(img_ir, img_p, output_pf)
        #
        # pd_grad = self.L_Grad1(img_p, img_pd, output_pf)
        #
        # p_grad_loss = self.grad(img_ir, img_p, img_pd, output_pf)
        #
        #
        # loss_all = p_loss + 0.5*p_grad_loss + 0.1*pd_grad     #0.5*p_loss + p_grad_loss +  + p_Inten_loss
        # # loss_all = p_loss + p_grad_loss + 0.5*pd_grad

        # loss_int, loss_grad =self.IGLoss(img_ir, img_p, output_pf)
        ssim_loss = self.mssim3_loss(img_ir, img_p, s1_s0, output_pf)
        # p_loss = self.mssim_loss(img_ir, img_p, output_pf)

        s0_loss = self.mae(x_s0, x_en_s0)
        dolp_loss = self.mae(x_dolp, x_en_dolp)
        en_de_loss = s0_loss + dolp_loss

        ploss = self.polar_loss(s0_de, dolp_de, aop, img_ir)


        loss_all = ssim_loss + 0.25*en_de_loss + 0.25*ploss


        # styleLoss = self.styleLoss(style, fusion_style)

        return loss_all  # self.mssim_loss(img_ir, img_p, output_pf)


if __name__ == '__main__':
    pass
