import os
import pickle

import numpy as np
import pandas as pd
import torch
# from distlib._backport import shutil
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
import glob
# import dataloader_functions as loaderFunc



# from dataloader_functions import files_preprocessing, get_data_analyse, data_preparing, create_dataframe_from_scalefile, \
#     find_max_content, create_dataframe_from_path

# from .dataloader_functions import files_preprocessing, get_data_analyse, data_preparing, \
#     create_dataframe_from_scalefile, \
#     find_max_content, create_df_from_csv


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


class CustomPictDataset(Dataset):
    ''' Custom class dataset for Calgary-Campinas Public Brain MR Dataset

        Attributes:
            df - datafraem with loaded scale table which consist next columns: ['name', 'mask', 'x', 'y', 'z']
            mark_dict - inform dict with scale information
            elem_dict - inform dict with context information
            self.amount - variable describing the number of pictures with information
            transform - torch transformation
            domain_choosen - boolean value which determines whether the domain has been selected
            data - dataframe with paths to image and mask

    '''

    def __init__(self, dataset_path, domain_mask, path_to_csv, mask_tail='_ss', save_dir='',
                 load_dir='', transform=transforms.Compose([transforms.Resize([int(300), int(300)]),
                                                            transforms.PILToTensor()]),
                 direct_load=False, path_to_csv_files=None, path_to_files_directory=None):
        """
        For first launch. Because of registred dataset, all main dataloader options are useless now. If u use directory
        with already registred images use 'direct_load' flag with path_to_csv_files or path_to_files_directory option.

        :param dataset_path: Absolute path to dataset images, for ex - 'Dataset/Original'.
        :param domain_mask: Absolute path to dataset mask,  for ex - 'Dataset/Mask'.
        :param path_to_csv: Absolute path to rescale csv file, for ex - './meta.csv'
        :param mask_tail: String parameter describing the tail by which the mask differs, '_ss' - default
        :param save_dir: Directory in which save all class value attributes (optional)
        :param load_dir: Directory from which Dataset will be loaded. IMPORTANT, if this parameter is specified,
        class will be loaded from it, without the preprocessing stage
        :param transform: Parameter with tensor transformation
        :param direct_load: Boolean flag, which indicate what we work with already registred images
        :param path_to_csv_files: Path to csv files with already created df. Type can be str if you want to analyse only
        one domain, or tuple of domains.
        :param path_to_files_directory: Path to csv files with already created df. Type can be str if you want to analyse only
        one domain, or tuple of domains.


        """

        #### Strange flex bcs our dataset already normalized. TODO make it normal
        if direct_load:
            # If we want to use multiply domains

            if path_to_csv_files and path_to_files_directory:
                print('Both paths are exceed. Using csv files option')

            if path_to_csv_files:
                path_to_csv_files = path_to_csv_files if type(path_to_csv_files) == tuple else (path_to_csv_files, )
                self.data = pd.concat((pd.read_csv(f) for f in path_to_csv_files))
            elif path_to_files_directory:
                path_to_files_directory = path_to_files_directory if type(path_to_files_directory) == tuple else \
                    (path_to_files_directory, )
                self.data = pd.concat((create_df_from_csv(f) for f in path_to_files_directory))
            else:
                raise ValueError('"direct_load" flag set to True, but path_to_csf_files and path_to_files_directory is '
                                 'empty. Set direct_load flag to False to recreate dataset, or ser path variables')
            self.domain_choosen = True
            self.transform = transform
            return

        if not load_dir:
            dataset_files = files_preprocessing(dataset_path[0], dataset_path[1], domain_mask, mask_tail)
            self.df = create_dataframe_from_scalefile(dataset_files, path_to_csv)
            self.mark_dict, self.elem_dict, self.file_res, self.amount = get_data_analyse(dataset_files)

            if save_dir:
                self.create_back(save_dir)
        else:
            self.__load_back(load_dir)
            self.mark_dict, self.elem_dict = {}, {}
            self.amount = find_max_content(self.file_res)

        self.transform = transform

        self.domain_choosen = False
        self.data = None

    def get_data_statistic(self):
        """
        :return: two dictionaries with statistics
        """
        return self.elem_dict, self.mark_dict

    def __load_back(self, load_dir):
        """
        This function download class important variables from directory

        :param load_dir: the directory from which the download takes place
        :return: None
        """

        with open(f'{load_dir}/file_inf.pickle', 'rb') as f:
            self.file_res = pickle.load(f)

        with open(f'{load_dir}/mark_inf.pickle', 'rb') as f:
            self.mark_dict = pickle.load(f)

        with open(f'{load_dir}/elem_inf.pickle', 'rb') as f:
            self.elem_dict = pickle.load(f)

        self.df = pd.read_csv(f'{load_dir}/scale_table.csv')
        self.df.set_index('id', inplace=True)

    def create_back(self, save_dir, rewrite_flag=False):
        """
        This function save class important variables in to directory

        :param save_dir: the directory from which the download takes place
        :param rewrite_flag: should this function overwrite files if there is such a directory
        :return: Nonec
        """

        if os.path.exists(save_dir):
            print(f'{save_dir} already exist.')

            if rewrite_flag:
                print('Rewriting directory')
            else:
                print('Stopping')
                return
        os.mkdir(save_dir)

        self.df.reset_index().to_csv(f'{save_dir}/scale_table.csv', index=False)

        with open(f'{save_dir}/file_inf.pickle', 'wb') as f:
            pickle.dump(self.file_res, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{save_dir}/mark_inf.pickle', 'wb') as f:
            pickle.dump(self.mark_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{save_dir}/elem_inf.pickle', 'wb') as f:
            pickle.dump(self.elem_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    ### TO DO REWRITE НАХУЙ
    def domain_preproc(self, path, domain_name, rewrite_flag=False):
        """

        :param path: Path to which all images of the given domain should be saved
        :param domain_name: domain name
        :param rewrite_flag: should this function overwrite files if there is such a directory
        :return:
        """
        if os.path.exists(path):
            print('This directory already exist')

            if rewrite_flag:
                print('Rewriting directory')
                # shutil.rmtree(path)
                data_preparing(path, self.file_res, self.amount, domain_name, self.df)
            else:
                print('Using existed directory.')
        else:
            data_preparing(path, self.file_res, self.amount, domain_name, self.df)

        # self.data = create_dataframe_from_path(path)
        self.domain_choosen = True

    def __len__(self):
        if self.domain_choosen:
            return len(self.data)
        else:
            print('You forget to choose current domain')

    def __getitem__(self, idx):
        if self.domain_choosen:
            row = self.data.iloc[idx]
            pict, mask = row['img'], row['mask']
            # test
            # brain = Image.open(pict).convert('L')
            # x_old,y_old = brain.size


            # x, y = 16 - (x_old%16), 16 - (y_old%16)
            # x_1, x_2 = (x///2, x//2) if x%2 == 0 else (x//2, x//2 + 1)
            # y_1, y_2 = (y//2, y//2) if y%2 == 0 else (y//2, y//2 + 1)


            # m = torch.nn.ZeroPad2d((x_1,x_2,y_1, y_2))
            #
            brain = self.transform(Image.open(pict).convert('L'))
            mask = self.transform(Image.open(mask).convert('L'))

            # brain, mask = m(brain), m(mask)

            mask = (mask > 200).to(mask.dtype)
            return {'image': brain, 'mask': mask}
        else:
            print('You forget to choose current domain')


class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        train_dir = root_dir + 'train/'
        mask_dir = root_dir + 'mask/'
        filenames = [y for x in os.walk(train_dir) for y in glob.glob(os.path.join(x[0], '*.npy'))]
        masks = [y for x in os.walk(mask_dir) for y in glob.glob(os.path.join(x[0], '*.npy'))]
        self.data = [(x,y) for x,y in zip(filenames, masks)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fpi, fpm = self.data[idx]
        img, mask = np.load(fpi).astype(np.float32), np.load(fpm).astype(np.float32)
        x_old,y_old = img.shape

        img, mask = torch.Tensor(img).unsqueeze(0), torch.Tensor(mask).unsqueeze(0)


        x, y = 16 - (x_old%16), 16 - (y_old%16)
        x_1, x_2 = (x//2, x//2) if x%2 == 0 else (x//2, x//2 + 1)
        y_1, y_2 = (y//2, y//2) if y%2 == 0 else (y//2, y//2 + 1)
        m = torch.nn.ZeroPad2d((y_1, y_2, x_1, x_2))

        # if self.transform:
        #     img, mask = self.transform(img), self.transform(mask)
        img, mask = m(img), m(mask)
        return {'image': img, 'mask': mask}

def tmp_save(file_list, domain_name):
    for file in file_list:
        current = np.load(file)
        filename = os.path.basename(file)[:-4]
        os.mkdir(f'/raid/data/DA_BrainDataset/dataset/{domain_name}/{filename}')
        for z in range(170):
            lvl = current[..., z]
            np.save(f'/raid/data/DA_BrainDataset/dataset/{domain_name}/{filename}/lvl{z}', lvl)


if __name__ == '__main__':
    path = '/raid/data/DA_BrainDataset/dataset/siemens3/'
    loader = BrainDataset(path)
    print(path)
