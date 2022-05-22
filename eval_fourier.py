import yaml
import os

import random
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter


from loader.dataloader import CustomPictDataset

from config_funcs import get_params
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


def eval_data(net, dataset, id_list):
    data_ids = dataset.data
    sdice_list = []
    net.eval()
    with torch.set_grad_enabled(False):
        for id in id_list:
            print(dataset.data)
            dataset.data = data_ids[data_ids == id]
            print(dataset.data)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            print(loader)
            data_list = []
            mask_list = []

            for i, batch in enumerate(tqdm(loader)):
                images, masks = batch['image'], batch['mask']
                images, masks = images.to(device).float(), masks.to(device).float()
                predicted_masks = net(images).sigmoid() ##Add sigmoid in forward
                predicted_masks = (predicted_masks > 0.9).to(int)

                data_list.append(predicted_masks)
                mask_list.append(masks)

            predict = torch.stack(data_list, dim=0)
            gr_truth = torch.stack(mask_list, dim=0)

            sdice_metric = lambda x, y: sdice(predict, gr_truth, (1, 1, 1), 1)
            print(sdice_metric)
            sdice_list.append(sdice_metric)

    # print(sdice_metric.mean())



def main():
    params = get_params('.')

    n_channels, image_size, batch_size = (params['n_channels'], params['image_size'], params['batch_size'])
    min_channels, max_channels, depth = (params['min_channels'], params['max_channels'], params['depth'])
    lr, n_epochs = (params['lr'], params['n_epochs'])
    backup_path, train_data_path, domain_name = (params['backup_dir'], params['domain_dir'], params['domain_name'])
    net = UNet2D(n_channels=n_channels, n_classes=1, init_features=min_channels, depth=depth, image_size=image_size[0]).to(device)

    net.load_state_dict(torch.load('siemens_3_basic.pth'))

    transform = transforms.Compose([transforms.Resize([int(image_size[0]), int(image_size[1])]),
                                    transforms.PILToTensor()])
    dataset = CustomPictDataset(None, None, None, direct_load=True, path_to_csv_files=os.path.join('/raid/data/DA_BrainDataset/ge15','df_save.csv'),
                                transform=transform)

    id_list = dataset.data.file_id.unique()

    eval_data(net, dataset, id_list)


    # data_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)


if __name__ == "__main__":
    main()