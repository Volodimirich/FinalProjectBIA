import yaml
import os

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from dataloader import CustomPictDataset
from torch.utils.data import DataLoader, random_split

from models import UNet2D, DenseNetSegmentation, ResNetSegmentation

from metrics import DiceMetric
from losses import CrossEntropyLoss, MultilabelDiceLoss, CombinedLoss, SoloClassDiceLoss

# for reproducibility
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice_loss(pred, target):
    epsilon = 1e-6
    inter = torch.dot(pred.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(pred) + torch.sum(target)
    if sets_sum.item() == 0:
        sets_sum = 2 * inter

    return (2 * inter + epsilon) / (sets_sum + epsilon)


def get_params(root):
    with open(os.path.join(root, "configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)
    params = {'train_data': configs['paths']['data']['train_data'],
              'dataset_table_path': configs['paths']['dataset_table'],
              'log_dir': configs['paths']['log_dir']}

    for param in params.keys():
        params[param] = os.path.join(root, params[param])

    params.update({'n_channels': int(configs['data_parameters']['n_channels']),
                   'image_size': tuple(map(int, configs['data_parameters']['image_size'].split(', '))),
                   'batch_size': int(configs['data_parameters']['batch_size'])})

    params.update({'init_features': int(configs['model_parameters']['UNet']['init_features']),
                   'depth': int(configs['model_parameters']['UNet']['depth'])})

    params.update({'num_init_features': int(configs['model_parameters']['DenseNet']['num_init_features']),
                   'growth_rate': int(configs['model_parameters']['DenseNet']['growth_rate']),
                   'block_config': tuple(map(int, configs['model_parameters']['DenseNet']['block_config'].split(', ')))})

    params.update({'blocks': tuple(map(int, configs['model_parameters']['ResNet']['blocks'].split(', ')))})

    params.update({'fourier_layer': configs['model_parameters']['Fourier']['fourier_layer']})

    params.update({'lr': float(configs['train_parameters']['lr']),
                   'n_epochs': int(configs['train_parameters']['epochs'])})

    return params


def make_dataset_table(data_path, csv_file_path):
    data_types = sorted([name for name in os.listdir(data_path)])
    image_paths = []
    for data_type in data_types:
        data_type_path = os.path.join(data_path, data_type)
        image_paths.extend(sorted([os.path.join(data_type_path, name) for name in os.listdir(data_type_path) if 'mask' not in name]))
    data = []

    print('dataset csv table creating...')
    for image_path in tqdm(image_paths):
        mask_path = ''.join([os.path.splitext(image_path)[0], '_mask.png'])
        data.append(np.array([image_path, mask_path]))

    pd.DataFrame(np.vstack(data), columns=['image', 'mask']).to_csv(csv_file_path, index=False)


def train_val_split(csv_file_path, val_size=0.2):
    dataset = pd.read_csv(csv_file_path)

    test_number = int(len(dataset) * val_size) + 1
    train_number = len(dataset) - test_number
    phase = ['train'] * train_number + ['val'] * test_number
    random.Random(1).shuffle(phase)

    pd.concat([dataset[['image', 'mask']],
               pd.DataFrame(phase, columns=['phase'])],
              axis=1).to_csv(csv_file_path, index=False)


def setup_experiment(title, params, log_dir):
    model_name = [title, 'fourier', params['fourier_layer']]
    model_name = '_'.join(model_name)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    best_model_path = f"{model_name}.best.pth"

    return writer, model_name, best_model_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_transform(batch_transform=None):
    def collate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        if batch_transform is not None:
            collated = batch_transform(collated)
        return collated
    return collate


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
          metric,
          n_epochs,
          device,
          writer,
          best_model_path):
    best_val_loss = float('+inf')
    for epoch in range(n_epochs):
        train_loss, train_metric = run_epoch(model, train_dataloader,
                                             criterion, optimizer,
                                             metric,
                                             phase='train', epoch=epoch,
                                             device=device, writer=writer)
        val_loss, val_metric = run_epoch(model, val_dataloader,
                                         criterion, None,
                                         metric,
                                         phase='val', epoch=epoch,
                                         device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Metric: {train_metric:.3f}')
        print(f'\t  Val Loss: {val_loss:.3f} |   Val Metric: {val_metric:.3f}')


def main():
    params = get_params('.')
    train_data_path, dataset_table_path, log_dir = (params['train_data'],
                                                    params['dataset_table_path'],
                                                    params['log_dir'])

    n_channels, image_size, batch_size = (params['n_channels'],
                                          params['image_size'],
                                          params['batch_size'])
    lr, n_epochs = (params['lr'], params['n_epochs'])


    dataset = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/BackY')
    dataset_val = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/BackY')
    dataset.domain_preproc('/raid/data/DA_BrainDataset/ge_3y', 'ge_3', val_amount=0.1)
    dataset_val.domain_preproc('/raid/data/DA_BrainDataset/ge_3y', 'ge_3')


    _, val_df = dataset.get_df()
    dataset_val.init_df(val_df)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)

    fourier_params = None
    if params['fourier_layer'] != 'None':
        fourier_params = {'fourier_layer': params['fourier_layer']}

    init_features, depth = params['init_features'], params['depth']
    model = UNet(n_channels=n_channels, n_classes=1,
                 init_features=init_features, depth=depth,
                 image_size=image_size, fourier_params=fourier_params).to(device)


    writer, model_name, best_model_path = setup_experiment(model.__class__.__name__, params, log_dir)
    best_model_path = os.path.join('.', best_model_path)
    print(f"Model name: {model_name}")
    print(f"Model has {count_parameters(model):,} trainable parameters")
    print()

    criterion = CombinedLoss([CrossEntropyLoss(), SoloClassDiceLoss()], [0.4, 0.6])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = DiceMetric()

    print("To see the learning process, use command in the new terminal:\ntensorboard --logdir <path to log directory>")
    print()
    train(model,
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