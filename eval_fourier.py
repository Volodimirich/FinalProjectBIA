import glob

import yaml
import os

import random
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from loader.dataloader import CustomPictDataset

from config_funcs import get_params
from loader.dataloader_functions import create_df_from_csv
from loses.loses import SoloClassDiceLossIvan, CombinedLoss, SoloClassDiceLoss
from metrics.metric_func import dice_loss, sdice
# for printing
from models.model import UNet2D

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_data(net, data):
    transform = transforms.Compose([transforms.Resize([int(256), int(256)]),
                                    transforms.PILToTensor()])
    net.eval()
    sdice_list = []
    with torch.no_grad():

        for i, (file_dir, mask_dir) in enumerate(data):
            data_list, mask_list = [], []
            for image, mask in zip(sorted(glob.glob(file_dir + '/*.png')), sorted(glob.glob(mask_dir + '/*.png'))):
                ##TODO rewrite fucking dataloader
                image = transform(Image.open(image).convert('L')).unsqueeze(0)
                mask = transform(Image.open(mask).convert('L')).unsqueeze(0)
                mask = mask > 200

                image, mask = image.to(device).float(), mask.to(device).float()
                predicted_masks = net(image).sigmoid()  ##Add sigmoid in forward
                predicted_masks = predicted_masks > 0.9

                data_list.append(predicted_masks.squeeze())
                mask_list.append(mask.squeeze())

            predict = torch.stack(data_list, dim=0)
            gr_truth = torch.stack(mask_list, dim=0)

            sdice_metric = sdice(predict.cpu().numpy(), gr_truth.cpu().numpy().astype(bool), (1, 1, 1), 1)
            print(sdice_metric)
            sdice_list.append(sdice_metric)

    print(sum(sdice_list) / len(sdice_list))


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
    net = UNet2D(n_channels=n_channels, n_classes=1, init_features=min_channels, depth=depth,
                 image_size=image_size[0]).to(device)

    net.load_state_dict(torch.load('siemens_3_basic.pth'))

    main_path = '/raid/data/DA_BrainDataset/ge15'
    data = get_dirs(main_path)
    eval_data(net, data)

    # data_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)


if __name__ == "__main__":
    main()
