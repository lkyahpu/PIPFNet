#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
# from model_TII import BiSeNet
# from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss, RegularizedLoss
# from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
from cv2.ximgproc import guidedFilter
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()



def train_fusion(num=0, logger=None):
    # num: control the segmodel 
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.cuda()
    # fusionmodel.load_state_dict(torch.load('model/Fusion/model_0.3045.pth'))
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)

    
    # train_dataset = Fusion_dataset('train')
    train_dataset = Fusion_dataset('train', dolp_path='dataset_all/train/dolp/', S0_path='dataset_all/train/S0/',
                                   s1_s0_path='dataset_all/train/s1_s0/', aop_path='dataset_all/train/aop/',
                                   label_path='dataset_all/train/label/')

    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    # 
    # if num>0:
    #     score_thres = 0.7
    #     ignore_idx = 255
    #     n_min = 8 * 640 * 480 // 8
        # criteria_p = OhemCELoss(
        #     thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        # criteria_16 = OhemCELoss(
        #     thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = RegularizedLoss()# Fusionloss()
    transform = transforms.Compose([transforms.ToTensor()])
    epoch = 200
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    loss_min = 10.0
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        prog_bar = tqdm(train_loader, desc="Epoch {}".format(epo+1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_S0, image_dolp, image_s1_s0,  image_aop, image_label, name) in enumerate(prog_bar):  #label,
            fusionmodel.train()
            # image_S0 = Variable(image_S0).cuda()
            # image_S0_ycrcb = RGB2YCrCb(image_S0)

            images_p_g = []
            for b in range(np.array(image_S0).shape[0]):
                img_p_g = guidedFilter(np.array(image_S0)[b, :, :, :].squeeze(),
                                       np.array(image_dolp)[b, :, :, :].squeeze(), 5,
                                       1e-5)
                img_p_g = Image.fromarray(img_p_g)
                img_p_g = transform(img_p_g)
                # img_p_g = torch.tensor(img_p_g).unsqueeze(0)
                images_p_g.append(img_p_g)
            images_p_g = torch.stack(images_p_g, axis=0)
            images_p_g = images_p_g.cuda()

            images_s1_s0_g = []
            for b in range(np.array(image_S0).shape[0]):
                img_s1_s0_g = guidedFilter(np.array(image_S0)[b, :, :, :].squeeze(),
                                       np.array(image_s1_s0)[b, :, :, :].squeeze(), 5,
                                       1e-5)
                img_s1_s0_g = Image.fromarray(img_s1_s0_g)
                img_s1_s0_g = transform(img_s1_s0_g)
                # img_s1_s0_g = torch.tensor(img_s1_s0_g).unsqueeze(0)
                images_s1_s0_g.append(img_s1_s0_g)
            images_s1_s0_g = torch.stack(images_s1_s0_g, axis=0)
            images_s1_s0_g = images_s1_s0_g.cuda()
            # images_p_g = images_p_g.double()


            image_S0 = image_S0.cuda()
            # image_dolp = Variable(image_dolp).cuda()
            image_dolp = image_dolp.cuda()

            image_aop = image_aop.cuda()

            image_label = image_label.cuda()

            # image_s1_s0 = image_s1_s0.cuda()




            # label = Variable(label).cuda()
            # logits = fusionmodel(image_S0, image_dolp)
            # fusion_ycrcb = torch.cat(
            #     (logits, image_S0_ycrcb[:, 1:2, :, :],
            #      image_S0_ycrcb[:, 2:, :, :]),
            #     dim=1,
            # )
            # fusion_image = YCrCb2RGB(fusion_ycrcb)

            fusion_image, s0_de, dolp_de, x_s0, x_dolp, x_en_s0, x_en_dolp = fusionmodel(image_S0, image_dolp)  #,style,fusion_style , x_s0, x_dolp, x_s0_en, x_dolp_en, s1_s0_en


            optimizer.zero_grad()
            # seg loss

            # fusion loss
            # loss_fusion, loss_in, loss_grad = criteria_fusion(
            #     image_S0_ycrcb, image_dolp, label, logits,num
            # )


            loss_fusion = criteria_fusion(image_S0, images_p_g, fusion_image, s0_de, dolp_de, images_s1_s0_g, image_aop,x_s0, x_dolp, x_en_s0, x_en_dolp)
            #,style,fusion_style ,x_s0, x_dolp, x_s0_en, x_dolp_en, s1_s0_en

            if num>0:
                loss_total = loss_fusion #+ (num) * seg_loss
            else:
                loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % train_loader.n_iter == 0:
                if num>0:
                    loss_seg=0
                else:
                    loss_seg=0
                msg = ', '.join(
                    [
                        'epoch: {epoch}',
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        # 'loss_in: {loss_in:.4f}',
                        # 'loss_grad: {loss_grad:.4f}',

                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]   #'loss_seg: {loss_seg:.4f}',
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    # loss_in=loss_in.item(),
                    # loss_grad=loss_grad.item(),
                    # loss_seg=loss_seg,
                    time=t_intv,
                    eta=eta,
                    epoch=epo+1,
                )
                logger.info(msg)
                st = ed
                if loss_total<loss_min:
                    loss_min = loss_total
                    fusion_model_file = os.path.join(modelpth, 'model_%.4f.pth'%loss_total)
                    torch.save(fusionmodel.state_dict(), fusion_model_file)
                    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
                    logger.info('\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    # modelpth = './model'
    # Method = 'Fusion'
    # modelpth = os.path.join(modelpth, Method)
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    train_fusion(logger=logger)
    # for i in range(4):
    #     train_fusion(i, logger)
    #     print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        # run_fusion('train')
        # print("|{0} Fusion Image Sucessfully~!".format(i + 1))
        # train_seg(i, logger)
        # print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")