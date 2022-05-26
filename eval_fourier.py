import glob

import yaml
import os

import random
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from dpipe.im.slices import iterate_slices
from dpipe.predict import divisible_shape, add_extract_dims
from dpipe.itertools import lmap
from tqdm import tqdm
from PIL import Image
from medpy.metric.binary import asd, hd

from dpipe.batch_iter import unpack_args
from torch.utils.data import DataLoader
from dpipe.torch import inference_step
from natsort import natsorted
from metrics.metric_func import dice_loss
import nibabel as nib

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from loader.dataloader import CustomPictDataset

from config_funcs import get_params
from loader.dataloader_functions import create_df_from_csv, scale_mri
from loses.loses import SoloClassDiceLossIvan, CombinedLoss, SoloClassDiceLoss
from metrics.metric_func import dice_loss, sdice
# for printing
from models.model import UNet2D
from models.modelIvan import UNet2D_harder
from models.segnet import SegNet

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper

SPATIAL_DIMS = (-3, -2, -1)

def eval_test(net, data):
    # file = '/raid/data/DA_BrainDataset/images/CC0247_ge_15_53_M.nii.gz'
    file = '/raid/data/DA_BrainDataset/images/CC0188_siemens_3_55_F.nii.gz'
    # mask = '/raid/data/DA_BrainDataset/masks/CC0247_ge_15_53_M_ss.nii.gz'
    mask = '/raid/data/DA_BrainDataset/masks/CC0188_siemens_3_55_F_ss.nii.gz'


    # predict
    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict(image):
        return inference_step(image, architecture=net, activation=torch.sigmoid)

    # print('Trouble here?')
    target = scale_mri(np.array(nib.load(file).dataobj).astype(np.float32)[..., :172])
    mask = scale_mri(np.array(nib.load(mask).dataobj).astype(np.float32)[..., :172]) > 0.5

    result = predict(target) > 0.5
    current_domain_sdice = sdice(result, mask, (1, 1, 1), 1)
    # hd_val = hd(result, mask)
    # asd_val = asd(result, mask)
    # print(hd_val)
    # print(asd_val)
    print(current_domain_sdice, 'from nii.gz')
    #
    # print(data[0])
    # img_2 = scale_mri(np.stack([np.asarray(Image.open(img).convert('L')).astype(np.float32) for img in natsorted(glob.glob(data[0] + '/*.png'))]), 2)
    # mask = scale_mri(np.stack([np.asarray(Image.open(img).convert('L')).astype(np.float32) for img in natsorted(glob.glob(data[1] + '/*.png'))]), 2) > 0.5
    # result_2 = predict(img_2) > 0.5

    # current_domain_sdice = sdice(result_2, mask, (1, 1, 1), 1)
    # print(current_domain_sdice, 'from img_space')

    img = np.rollaxis(np.load('/raid/data/DA_BrainDataset/siemens_3_np/CC0188_siemens_3_55_F.npy').astype(np.float32)[..., :170],2)
    img -= img.min()
    print(img.min(), img.max())
    img /= img.max()
    # img *= 255
    print(img.min(), img.max())

    mask = np.rollaxis(np.load('/raid/data/DA_BrainDataset/siemens_3_np_mask/CC0188_siemens_3_55_F.npy').astype(bool)[..., :170],2)

    result_3 = predict(img) > 0.5
    current_domain_sdice = sdice(result_3, mask, (1, 1, 1), 1)
    print(current_domain_sdice, 'from npy_space')

    # sdice_metric = sdice(predict.cpu().numpy().astype(bool), gr_truth.cpu().numpy().astype(bool), (1, 1, 1), 1)


def eval_data(net, data):
    transform = transforms.Compose([transforms.Resize([int(256), int(256)]),
                                    transforms.PILToTensor()])
    net.eval()
    sd_list = []
    hd_list = []
    asd_list = []

    with torch.no_grad():

        for i, (file_dir, mask_dir) in enumerate(data):
            data_list, mask_list = [], []
            for image, mask in zip(natsorted(glob.glob(file_dir + '/*.png')), natsorted(glob.glob(mask_dir + '/*.png'))):
                ##TODO rewrite fucking dataloader

                image = transform(Image.open(image).convert('L')).unsqueeze(0)
                mask = transform(Image.open(mask).convert('L')).unsqueeze(0)
                mask = (mask > 200).to(mask.dtype)

                # predict
                @slicewise  # 3D -> 2D iteratively
                @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
                @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
                def predict(image):
                    return inference_step(image, architecture=net, activation=torch.sigmoid)

                image, mask = image.to(device).float(), mask.to(device).float()
                predicted_masks = net(image).sigmoid()  ##Add sigmoid in forward
                predicted_masks = predicted_masks > 0.9

                data_list.append(predicted_masks.squeeze())
                mask_list.append(mask.squeeze())

            predict = torch.stack(data_list, dim=0)
            gr_truth = torch.stack(mask_list, dim=0)

            sdice_metric = sdice(predict.cpu().numpy().astype(bool), gr_truth.cpu().numpy().astype(bool), (1, 1, 1), 1)
            hd_val = hd(predict.cpu().numpy(), gr_truth.cpu().numpy())
            asd_val= asd(predict.cpu().numpy(), gr_truth.cpu().numpy())
            print(sdice_metric)
            print(hd_val)
            print(asd_val)
            return

            sd_list.append(sdice_metric)
            hd_list.append(hd_val)
            asd_list.append(asd_val)


    print(sum(sdice_metric) / len(sdice_metric))
    print(sum(hd_list) / len(hd_list))
    print(sum(asd_list) / len(asd_list))


def get_dirs(root_path):
    all_images = glob.glob(f'{root_path}/' + 'train/*')
    all_masks = glob.glob(f'{root_path}/' + 'mask/*')
    result = []

    ##Maybe Pathlib + replace is better
    for image in all_images:
        mask = next(x for x in all_masks if os.path.basename(image) in x)
        result.append((image, mask))

    return result


def main():
    params = get_params('.')

    n_channels, image_size, batch_size = (params['n_channels'], params['image_size'], params['batch_size'])
    min_channels, max_channels, depth = (params['min_channels'], params['max_channels'], params['depth'])
    # net = UNet2D(n_channels=n_channels, n_classes=1, init_features=min_channels, depth=depth,
    #              image_size=image_size[0]).to(device)
    net = UNet2D_harder(n_chans_in=n_channels, n_chans_out=1, n_filters_init=16).to(device)
    # net = SegNet(n_classes=1).to(device)


    net.load_state_dict(torch.load('model_1.pth'))

    main_path = '/raid/data/DA_BrainDataset/ge_15'
    data = get_dirs(main_path)
    # data = (data[0],)
    data = ['/raid/data/DA_BrainDataset/ge15/train/CC0280_ge_15_57_F/', '/raid/data/DA_BrainDataset/ge15/mask/CC0280_ge_15_57_F/']
    # eval_data(net, data)
    # eval_data_test(net, data)
    eval_test(net, data)
    # data_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)


if __name__ == "__main__":
    main()
