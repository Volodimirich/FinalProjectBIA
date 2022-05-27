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
from medpy.metric.binary import dc, hd
import sys

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
from skimage.transform import resize

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper


def send_images_to_tensorboard(writer, data, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = torchvision.utils.make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)

SPATIAL_DIMS = (-3, -2, -1)

def eval(net, data, writer, from_baselines=False, debug=False):
    # predict
    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict(image):
        return inference_step(image, architecture=net, activation=torch.sigmoid)

    sd_list, hd_list, asd_list = [], [], []

    for i, (file_dir, mask_dir) in enumerate(tqdm(data)):

        img = np.load(file_dir).astype(np.float32)
        if not from_baselines:
            img = np.rollaxis(img[..., :170],2)
        else:
            img = resize(img, (170, 182, 218))


        img -= img.min()
        img /= img.max()

        mask = np.rollaxis(np.load(mask_dir).astype(bool)[..., :170],2)
        #
        result = predict(img) > 0.9
        current_domain_sdice = sdice(result, mask, (1, 1, 1), 2)


        hd_val = hd(result, mask)
        dc_val = dc(result, mask)

        if debug:
            print(current_domain_sdice, 'sdice')
            print(hd_val, 'hd')
            print(dc_val, 'dc')


        sd_list.append(current_domain_sdice)
        hd_list.append(hd_val)
        asd_list.append(dc_val)

        if writer is not None:
            send_images_to_tensorboard(writer, torch.from_numpy(result)[...,50].float() * 255, name='predict', iter=i)
            send_images_to_tensorboard(writer, torch.from_numpy(mask)[...,50].float() * 255, name='train', iter=i)


    print(sum(sd_list) / len(sd_list))
    print(sum(hd_list) / len(hd_list))
    print(sum(asd_list) / len(asd_list))


def get_dirs(train_dir, mask_dir):
    all_images = glob.glob(f'{train_dir}/*')
    all_masks = glob.glob(f'{mask_dir}/*')
    result = []

    ##Maybe Pathlib + replace is better
    for image in all_images:
        mask = next(x for x in all_masks if os.path.basename(image) in x)
        result.append((image, mask))

    return result


def main():
    params = get_params('.')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    n_channels, image_size, batch_size = (params['n_channels'], params['image_size'], params['batch_size'])
    min_channels, max_channels, depth = (params['min_channels'], params['max_channels'], params['depth'])

    if len(sys.argv) != 1:
        dev = int(sys.argv[1])
        if dev < 0 or dev > 3:
            print('Using incorrect device, basic start')
        else:
            print(dev)
            device = torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")


    # net = UNet2D(n_channels=n_channels, n_classes=1, init_features=min_channels, depth=depth,
    #              image_size=image_size[0]).to(device)
    net = UNet2D_harder(n_chans_in=n_channels, n_chans_out=1, n_filters_init=16).to(device)
    # net = SegNet(n_classes=1).to(device)


    # net.load_state_dict(torch.load('siemens3_basic.pth')['model_state_dict'])
    # if dev == 0:
    net.load_state_dict(torch.load('weights/model_5.pth'))
    # elif dev == 1:
    #     net.load_state_dict(torch.load('siemens15_basic_5.pth'))
    # elif dev == 2:
    #     net.load_state_dict(torch.load('siemens15_basic_10.pth'))
    # elif dev == 3:
    #     net.load_state_dict(torch.load('siemens15_basic_15.pth'))
    # net.load_state_dict(torch.load('ge15_10_padding_ivan.pth'))

    # train_dir = '/raid/data/DA_BrainDataset/siemens_3_np'

    # train_dir = '/raid/data/DA_BrainDataset/ge15CUT'
    mask_dir = '/raid/data/DA_BrainDataset/ge_15_np_mask'

    # train_dir = '/raid/data/DA_BrainDataset/ph3CUT'
    # mask_dir = '/raid/data/DA_BrainDataset/siemens_15_np_mask'

    ### debug
    # train_data = ['CC0240_ge_15_60_F.npy', 'CC0241_ge_15_60_F.npy', 'CC0242_ge_15_55_F.npy', 'CC0243_ge_15_58_F.npy',
    #               'CC0244_ge_15_56_F.npy', 'CC0245_ge_15_48_M.npy']
    train_data = ['CC0120_siemens_15_58_F', 'CC0121_siemens_15_61_M', 'CC0122_siemens_15_53_F', 'CC0123_siemens_15_58_M',
                  'CC0124_siemens_15_57_F', 'CC0125_siemens_15_65_F']
    data = [('/raid/data/DA_BrainDataset/Vladimir_base/siemens_15_np_to_siemens_3_np/' + path, mask_dir + '/' + path) for path in train_data]
    print(data)
    # data = get_dirs(train_dir, mask_dir)
    writer = SummaryWriter()

    print('cut')
    eval(net, data, writer, from_baselines=True, debug=True)


if __name__ == "__main__":
    main()
