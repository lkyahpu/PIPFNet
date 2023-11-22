# coding:utf-8
import os
import argparse
import time
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from model_TII import BiSeNet
from TaskFusion_dataset import Fusion_dataset
from FusionNet import FusionNet
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image


# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main():
    fusion_model_path = './model.pth'
    fusionmodel = eval('FusionNet')(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        fusionmodel.to(device)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')

    dolp_path = 'dataset_all/test/dolp/'
    S0_path = 'dataset_all/test/S0/'
    label_path = 'dataset_all/test/label/'

    test_len = len(os.listdir(S0_path))
    test_dataset = Fusion_dataset('val', S0_path=S0_path, dolp_path=dolp_path, label_path=label_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # test_loader.n_iter = len(test_loader)
    time_sum = 0

    with torch.no_grad():
        for it, (images_s0, images_dolp, image_label,name) in enumerate(test_loader):
            # images_vis = Variable(images_vis)
            # images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_s0 = images_s0.to(device)
                images_dolp = images_dolp.to(device)
            # images_vis_ycrcb = RGB2YCrCb(images_vis)
            # torch.cuda.synchronize()
            time_st = time.time()
            fusion_image,_,_,_,_,_,_ = fusionmodel(images_s0, images_dolp)
            # torch.cuda.synchronize()
            # fusion_image.cpu()
            time_ed = time.time()
            time_sum += time_ed-time_st
            # fusion_ycrcb = torch.cat(
            #     (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
            #     dim=1,
            # )
            # fusion_image = YCrCb2RGB(fusion_ycrcb)
            # ones = torch.ones_like(fusion_image)
            # zeros = torch.zeros_like(fusion_image)
            # fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            # fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))

            #
            # fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, 0]
                # image = np.power(image, 0.7)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))

                image = np.uint8(255.0 * image)
                image = Image.fromarray(image)

                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

    print("Average %.3f frames per second"%(test_len/time_sum))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='IPF')
    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=2)
    args = parser.parse_args()
    n_class = 9
    # seg_model_path = './model/Fusion/model_final.pth'
    # fusion_model_path = './model/Fusion/fusionmodel_final.pth'
    fused_dir = 'Fusion_results'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
