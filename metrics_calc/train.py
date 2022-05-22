import os

import random
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import torch

from torch.utils.tensorboard import SummaryWriter


from loader.dataloader import CustomPictDataset
from loses.loses import CombinedLoss, SoloClassDiceLoss
from metrics.metric import SDiceMetric
# for printing
from models.modelIvan import UNet2D

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

    epoch_loss = 0.0
    epoch_metric = 0.0

    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(tqdm(iterator)):
            images, masks = batch['image'], batch['mask']
            images, masks = images.to(device).float(), masks.to(device).float()
            predicted_masks = model(images).sigmoid()

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

    net = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)
    transform = transforms.Compose([transforms.Resize([int(256), int(256)]),
                                    transforms.PILToTensor()])
    dataset = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/BackX', transform = transform)
    dataset_val = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/BackX', transform = transform)

    #AbsPath is better
    dataset.domain_preproc('/raid/data/DA_BrainDataset/siemens_3x', 'siemens_3', val_amount=0.1)
    dataset_val.domain_preproc('/raid/data/DA_BrainDataset/siemens_3x', 'siemens_3')


    _, val_df = dataset.get_df()
    dataset_val.init_df(val_df)

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=8, shuffle=True, drop_last=True)


    criterion = CombinedLoss([CrossEntropyLoss(), SoloClassDiceLoss()], [0.8, 0.2])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = SDiceMetric()
    n_epochs = 5
    writer = SummaryWriter(log_dir=os.path.join('.'))
    best_model_path = f"Siemens3_x.basic.pth"

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