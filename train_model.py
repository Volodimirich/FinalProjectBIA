import yaml
import os

import random
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm

from dpipe.predict import divisible_shape
from dpipe.torch import inference_step

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter


from loader.dataloader import CustomPictDataset

from config_funcs import get_params
from loses.loses import SoloClassDiceLossIvan, CombinedLoss, SoloClassDiceLoss
from metrics.metric_func import dice_loss
# for printing
from models.model import UNet2D
from models.modelIvan import UNet2D_harder
from models.segnet import SegNet

torch.set_printoptions(precision=2)

# for reproducibility
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPATIAL_DIMS = (-3, -2, -1)


def send_images_to_tensorboard(writer, data, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid =  torchvision.utils.make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)



def run_epoch(model, iterator,
              criterion, optimizer,
              metric,
              phase='train', epoch=0,
              device='cpu', writer=None):

    is_train = (phase == 'train')


    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss, epoch_metric = 0, 0

    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(tqdm(iterator)):
            images, masks = batch['image'], batch['mask']
            images, masks = images.to(device).float(), masks.to(device).float()
            predicted_masks = model(images).sigmoid() ##Add sigmoid in forward


            loss = criterion(predicted_masks, masks)
            # loss = criterion(predicted_masks.float(), masks.float()) + dice_loss(predicted_masks, masks.float()).float()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            predicted_masks = (predicted_masks > 0.9).to(int)
            epoch_metric += metric(predicted_masks.float(), masks)


        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / len(iterator), epoch)
            writer.add_scalar(f"metric_epoch/{phase}", epoch_metric / len(iterator), epoch)

            send_images_to_tensorboard(writer, predicted_masks.float(), iter=epoch, name='predict')
            send_images_to_tensorboard(writer, masks.float(), iter=epoch, name='mask')

            # writer.add_images(f'predict/{phase}', predicted_masks, dataformats='NCHW')
            # writer.add_images(f'masks/{phase}', masks, epoch, dataformats='NCHW')


        return epoch_loss / len(iterator), epoch_metric / len(iterator)


def train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metrics,
          n_epochs,
          device,
          writer,
          best_model_path):
    best_val_loss = float('+inf')


    for epoch in range(n_epochs):
        train_loss, train_metric = run_epoch(model, train_dataloader,
                                              criterion, optimizer,
                                              metrics,
                                              phase='train', epoch=epoch,
                                              device=device, writer=writer)
        val_loss, val_metric = run_epoch(model, val_dataloader,
                                          criterion, None,
                                          metrics,
                                          phase='val', epoch=epoch,
                                          device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch: {epoch + 1:02}')

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Metric: {train_metric:.3f}')
        print(f'\t  Val Loss: {val_loss:.3f} |   Val Metric: {val_metric:.3f}')


def main():
    params = get_params('.')

    n_channels, image_size, batch_size = (params['n_channels'], params['image_size'], params['batch_size'])
    min_channels, max_channels, depth = (params['min_channels'], params['max_channels'], params['depth'])
    lr, n_epochs = (params['lr'], params['n_epochs'])
    train_data_path, domain_name = (params['domain_dir'], params['domain_name'])
    # net = UNet2D(n_channels=n_channels, n_classes=1, init_features=min_channels, depth=depth, image_size=image_size[0]).to(device)
    net = UNet2D_harder(n_chans_in=n_channels, n_chans_out=1, n_filters_init=16).to(device)
    # net = SegNet(n_classes=1).to(device)
    transform = transforms.Compose([transforms.Resize([int(image_size[0]), int(image_size[1])]),
                                    transforms.PILToTensor()])

    # transform = transforms.Compose([transforms.PILToTensor()])
    # dataset = CustomPictDataset(None, None, None, direct_load=True, path_to_csv_files=os.path.join(train_data_path,'df_save.csv'),
    #                             transform=transform)

    dataset = CustomPictDataset(None, None, None, direct_load=True, path_to_files_directory=train_data_path, transform=transform)

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))


    data_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, drop_last=True)


    criterion = CombinedLoss([CrossEntropyLoss(), SoloClassDiceLoss()], [0.8, 0.2])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = dice_loss
    n_epochs = 5
    writer = SummaryWriter()
    # writer=None
    best_model_path = f"{domain_name}_padding_ivan.pth"

    print("To see the learning process, use command in the new terminal:\ntensorboard --logdir <path to log directory>")
    print()
    train(net,
          data_loader, val_loader,
          criterion,
          optimizer, scheduler,
          metric,
          n_epochs,
          device,
          writer,
          best_model_path)




if __name__ == "__main__":
    main()