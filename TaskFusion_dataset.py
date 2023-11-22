# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
import torchvision.transforms as transforms
import cv2

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, dolp_path = None, S0_path = None, s1_s0_path = None, aop_path = None, label_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val'], 'split must be "train"|"val"|"test"'
        # self.transform_train = transforms.Compose(
        #     [transforms.Resize((128, 160)), transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5),
        #       transforms.RandomRotation(degrees=5)])  #transforms.RandomRotation(degrees=15),

        self.transform_train = transforms.Compose(
            [transforms.Resize((128, 160)), transforms.ToTensor()])  # transforms.RandomRotation(degrees=15),


        self.transform_test = transforms.Compose(
            [transforms.ToTensor()])


        if split == 'train':
            data_dir_S0    = S0_path
            data_dir_dolp  = dolp_path  #'dataset_all/train/dolp/'
            data_dir_s1_s0 = s1_s0_path #'dataset_all/train/s1_s0/'
            data_dir_aop   = aop_path   #'dataset_all/train/aop/'
            data_dir_label = label_path #'dataset_all/train/label/'
            # data_dir_label = './MSRS/Label/train/MSRS/'

            self.filepath_S0, self.filenames_S0 = prepare_data_path(data_dir_S0)
            self.filepath_dolp, self.filenames_dolp = prepare_data_path(data_dir_dolp)
            self.filepath_s1_s0, self.filenames_s1_s0 = prepare_data_path(data_dir_s1_s0)
            self.filepath_aop, self.filenames_aop = prepare_data_path(data_dir_aop)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)

            self.split = split
            self.length = min(len(self.filenames_S0), len(self.filenames_dolp))

        elif split == 'val':
            data_dir_S0 = S0_path
            data_dir_dolp = dolp_path
            data_dir_label = label_path

            self.filepath_S0, self.filenames_S0 = prepare_data_path(data_dir_S0)
            self.filepath_dolp, self.filenames_dolp = prepare_data_path(data_dir_dolp)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)

            self.split = split
            self.length = min(len(self.filenames_S0), len(self.filenames_dolp))

        else:
            data_dir_S0 = S0_path
            data_dir_dolp = dolp_path
            data_dir_label = label_path

            self.filepath_S0, self.filenames_S0 = prepare_data_path(data_dir_S0)
            self.filepath_dolp, self.filenames_dolp = prepare_data_path(data_dir_dolp)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)

            self.split = split
            self.length = min(len(self.filenames_S0), len(self.filenames_dolp))


    def __getitem__(self, index):
        if self.split=='train':
            S0_path = self.filepath_S0[index]
            dolp_path = self.filepath_dolp[index]
            s1_s0_path = self.filepath_s1_s0[index]
            aop_path = self.filepath_aop[index]
            label_path = self.filepath_label[index]
            # image_S0 = np.array(Image.open(S0_path))

            # image_S0 = cv2.imread(S0_path, 0)
            # image_dolp = cv2.imread(dolp_path, 0)
            image_S0 = Image.open(S0_path)
            image_dolp = Image.open(dolp_path)
            image_s1_s0 = Image.open(s1_s0_path)
            image_aop = Image.open(aop_path)

            image_label = Image.open(label_path)
            # image_label = cv2.imread(label_path, 0)
            #
            # if image_label.max() > 1:
            #     image_label = image_label / 255
            # image_label = Image.fromarray(np.uint8(image_label))


            image_S0 = self.transform_train(image_S0).float()
            image_dolp = self.transform_train(image_dolp).float()
            image_s1_s0 = self.transform_train(image_s1_s0).float()
            image_aop = self.transform_train(image_aop).float()
            image_label = self.transform_train(image_label).float()
            # image_label[image_label > 0] = 1



            # label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_S0[index]
            return (
                image_S0,
                image_dolp,
                image_s1_s0,
                image_aop,
                image_label,
                #torch.tensor(label),
                name,
            )

        elif self.split=='val':
            S0_path = self.filepath_S0[index]
            dolp_path = self.filepath_dolp[index]
            label_path = self.filepath_label[index]

            image_S0 = Image.open(S0_path)
            image_dolp = Image.open(dolp_path)
            image_label = Image.open(label_path)

            # image_label = cv2.imread(label_path, 0)
            #
            # if image_label.max() > 1:
            #     image_label = image_label / 255
            # image_label = Image.fromarray(np.uint8(image_label))

            image_S0 = self.transform_test(image_S0).float()
            image_dolp = self.transform_test(image_dolp).float()
            image_label = self.transform_test(image_label).float()
            # image_label[image_label > 0] = 1

            # image_S0 = image_S0.unsqueeze(0)
            # image_dolp = image_dolp.unsqueeze(0)

            name = self.filenames_S0[index]
            return (
                image_S0,
                image_dolp,
                image_label,
                name,
            )
        else:
            S0_path = self.filepath_S0[index]
            dolp_path = self.filepath_dolp[index]
            label_path = self.filepath_label[index]

            image_S0 = Image.open(S0_path)
            image_dolp = Image.open(dolp_path)
            image_label = Image.open(label_path)

            image_S0 = self.transform_test(image_S0).float()
            image_dolp = self.transform_test(image_dolp).float()
            image_label = self.transform_test(image_label).float()

            name = self.filenames_S0[index]
            return (
                image_S0,
                image_dolp,
                image_label,
                name,
            )

    def __len__(self):
        return self.length

# if __name__ == '__main__':
    # data_dir = '/data1/yjt/MFFusion/dataset/'
    # train_dataset = MF_dataset(data_dir, 'train', have_label=True)
    # print("the training dataset is length:{}".format(train_dataset.length))
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # train_loader.n_iter = len(train_loader)
    # for it, (image_S0, image_dolp, label) in enumerate(train_loader):
    #     if it == 5:
    #         image_S0.numpy()
    #         print(image_S0.shape)
    #         image_dolp.numpy()
    #         print(image_dolp.shape)
    #         break
