import yaml
import argparse
import numpy as np
import os
import glob
import torch
from torchvision import transforms
import torch.nn.functional as F

from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
import torch.backends.cudnn as cudnn
from models import *
from PIL import Image


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])


path = './savedmodels/VanillaVAE/VanillaVAE_s64.pt'
model.load_state_dict(torch.load(path))
model = model.to('cuda:0')
model.eval()

batch_size = 144*16
subdir = 'vae1/'

with torch.no_grad():
    for j in range(611):# Give it a random variable
        print(f'{j+1}/1000')
        z = torch.randn(batch_size, model.latent_dim)
        z = z.to('cuda:0')
        samples = model.decode(z)
        samples = F.interpolate(samples, (32,32), mode='bilinear')
        for i in range(batch_size):
            # samples[i] = samples[i].unsqueeze(1)
            # print(samples[i].shape)
            # samples[i] = transforms.Resize(32)(samples[i])
            # samples[i] = samples[i].squeeze(1)
            im = np.uint8(samples[i].detach().cpu().numpy() * 255)
            im = np.squeeze(im, axis=0)
            im = Image.fromarray(im, mode='L')
            im.save('savedtiles/' + subdir + 'tile_' + str(j) + '_' + str(i) + '.png', mode='L')