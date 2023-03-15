# %%
import os

os.chdir('/Users/venkat/Documents/EDSR-PyTorch/src')
from sklearn.utils import Bunch
import torch

import utility
import data
import model
import loss
from trainer import Trainer
from model import edsr

# %%
default_arg = {
    'debug': True,
    'template': '.',
    'n_threads': 6,
    'cpu': True,
    'n_GPUs': 1,
    'seed': 1,
    'dir_data': '../../../dataset',
    'dir_demo': '../test',
    'data_train': 'DIV2K',
    'data_test': 'DIV2K',
    'data_range': '1-800/801-810',
    'ext': 'sep',
    'scale': 4,
    'patch_size': 192,
    'rgb_range': 255,
    'n_colors': 3,
    'chop': True,
    'no_augment': True,
    'model': 'EDSR',
    'act': 'relu',
    'pre_train': '',
    'extend': '.',
    'n_resblocks': 16,
    'n_feats': 64,
    'res_scale': 1,
    'shift_mean': True,
    'dilation': True,
    'precision': 'single',
    'G0': 64,
    'RDNkSize': 3,
    'RDNconfig': 'B',
    'n_resgroups': 10,
    'reduction': 16,
    'reset': True,
    'test_every': 1000,
    'epochs': 300,
    'batch_size': 16,
    'split_batch': 1,
    'self_ensemble': True,
    'test_only': True,
    'gan_k': 1,
    'lr': 1e-4,
    'decay': '200',
    'gamma': 0.5,
    'optimizer': 'ADAM',
    'momentum': 0.9,
    'betas': (0.9, 0.999),
    'epsilon': 1e-8,
    'weight_decay': 0,
    'gclip': 0,
    'loss': "1*L1",
    'skip_threshold': 1e8,
    'save': 'test',
    'load': '',
    'resume': 0,
    'save_models': True,
    'print_every': 100,
    'save_results': True,
    'save_gt': True
}

# %%
args = default_arg | {
    'model': 'MDSR',
    'data_test': ['Demo'],
    'scale': [2, 3, 4],
    'pre_train': 'download',
    'save_results': True,
    'self_ensemble': False,
    'chop': False,
    'save': 'second_try',
    'n_threads': 1
}

# %%

args = Bunch(**args)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)



# %%
if __name__ == '__main__':
    
    loader = data.Data(args)
    scale = args.scale
    idx_scale = 0
    self_ensemble = args.self_ensemble
    cpu = args.cpu
    device = torch.device('cpu' if args.cpu else 'cuda')
    n_GPUs = args.n_GPUs
    model = edsr.make_model(args).to(device)


    ckp = checkpoint
    apath = ckp.get_path('model')
    kwargs = {'map_location': lambda storage, loc: storage}

    print('Download the model')
    dir_model = os.path.join('..', 'models')
    os.makedirs(dir_model, exist_ok=True)
    load_from = torch.utils.model_zoo.load_url(
        model.url,
        model_dir=dir_model,
        **kwargs
    )
    model.load_state_dict(load_from, strict=False)
    print(model, file=ckp.log_file)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, _loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

