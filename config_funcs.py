import os
import yaml

def get_params(root):
    with open(os.path.join(root, "configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)
    params = {'domain_dir': configs['paths']['domain_dir'],
              'domain_name': configs['paths']['domain_name'],
              'model_name': configs['paths']['model_name']}

    for param in params.keys():
        params[param] = os.path.join(root, params[param])


    params.update({'n_channels': int(configs['data_parameters']['n_channels']),
                   'image_size': tuple(map(int, configs['data_parameters']['image_size'].split(', '))),
                   'batch_size': int(configs['data_parameters']['batch_size']),
                   'ratio': float(configs['data_parameters']['ratio'])})

    params.update({'min_channels': int(configs['model_parameters']['UNet']['min_channels']),
                   'max_channels': int(configs['model_parameters']['UNet']['max_channels']),
                   'depth': int(configs['model_parameters']['UNet']['depth']),
                   })


    params.update({'lr': float(configs['train_parameters']['lr']),
                   'n_epochs': int(configs['train_parameters']['epochs'])})

    return params