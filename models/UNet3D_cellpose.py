# -*- coding: utf-8 -*-
"""
# 3D Cellpose Extension.
# Copyright (C) 2021 D. Eschweiler, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please refer to the documentation for more information about the software
# as well as for installation instructions.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataloader.h5_dataloader import MeristemH5Dataset
from ThirdParty.radam import RAdam
from models.UNet3D import UNet3D_module

    
    
    
class UNet3D_cellpose(pl.LightningModule):
    
    def __init__(self, hparams):
        super(UNet3D_cellpose, self).__init__()
        
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.augmentation_dict = {}

        # networks
        self.network = UNet3D_module(patch_size=hparams.patch_size, in_channels=hparams.in_channels, out_channels=hparams.out_channels, feat_channels=hparams.feat_channels, out_activation=hparams.out_activation, norm_method=hparams.norm_method)

        # cache for generated images
        self.last_predictions = None
        self.last_imgs = None
        self.last_masks = None


    def forward(self, z):
        return self.network(z)
    
    
    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        
        if isinstance(pretrained_file, (list,tuple)):
            pretrained_file = pretrained_file[0]
        
        # Load the state dict
        state_dict = torch.load(pretrained_file)['state_dict']
        
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            
        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())
        
        layers = []
        for layer in param_dict:
            if strict and not 'network.'+layer in state_dict:
                if verbose:
                    print('Could not find weights for layer "{0}"'.format(layer))
                continue
            try:
                param_dict[layer].data.copy_(state_dict['network.'+layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print('Error at layer {0}:\n{1}'.format(layer, e))
        
        self.network.load_state_dict(param_dict)
        
        if verbose:
            print('Loaded weights for the following layers:\n{0}'.format(layers))
            

    def background_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    
    def flow_loss(self, y_hat, y, mask):
        loss = F.mse_loss(y_hat, y, reduction='none')
        weight = torch.clamp(mask, min=0.01, max=1.0)
        loss = torch.mul(loss, weight)    
        loss = torch.sum(loss)
        loss = torch.div(loss, torch.clamp(torch.sum(weight), 1, mask.numel())) 
        return loss


    def training_step(self, batch, batch_idx):
        
        # Get image ans mask of current batch
        self.last_imgs, self.last_masks = batch['image'], batch['mask']
        
        # generate images
        self.predictions = self.forward(self.last_imgs)
                
        # get the losses
        loss_bg = self.background_loss(self.predictions[:,0,...], self.last_masks[:,0,...])
        
        loss_flowx = self.flow_loss(self.predictions[:,1,...], self.last_masks[:,1,...], self.last_masks[:,0,...])
        loss_flowy = self.flow_loss(self.predictions[:,2,...], self.last_masks[:,2,...], self.last_masks[:,0,...])
        loss_flowz = self.flow_loss(self.predictions[:,3,...], self.last_masks[:,3,...], self.last_masks[:,0,...])
        loss_flow = (loss_flowx + loss_flowy + loss_flowz)/3 
        
        loss = self.hparams.background_weight * loss_bg + \
               self.hparams.flow_weight * loss_flow
        tqdm_dict = {'bg_loss': loss_bg, 'flow_loss':loss_flow, 'epoch': self.current_epoch}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output
        
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)
        return {'test_loss': F.mse_loss(y_hat, y)} 

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)
        return {'val_loss': F.mse_loss(y_hat, y)} 

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = RAdam(self.network.parameters(), lr=self.hparams.learning_rate)
        return [opt], []

    def train_dataloader(self):
         if self.hparams.train_list is None:
            return None
         else:
            dataset = MeristemH5Dataset(self.hparams.train_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict=self.augmentation_dict,\
                                        dist_handling=self.hparams.dist_handling, norm_method=self.hparams.data_norm, sample_per_epoch=self.hparams.samples_per_epoch)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def test_dataloader(self):
        if self.hparams.test_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.test_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling=self.hparams.dist_handling, norm_method=self.hparams.data_norm)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        if self.hparams.val_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.val_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling=self.hparams.dist_handling, norm_method=self.hparams.data_norm)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_epoch_end(self):
        
        # log sampled images
        predictions = self.forward(self.last_imgs)
        prediction_grid = torchvision.utils.make_grid(predictions[0,:,np.newaxis,int(self.hparams.patch_size[0]//2),:,:])
        self.logger.experiment.add_image('generated_images', prediction_grid, self.current_epoch)
        
        img_grid = torchvision.utils.make_grid(self.last_imgs[0,:,np.newaxis,int(self.hparams.patch_size[0]//2),:,:])
        self.logger.experiment.add_image('raw_images', img_grid, self.current_epoch)
        
        mask_grid = torchvision.utils.make_grid(self.last_masks[0,:,np.newaxis,int(self.hparams.patch_size[0]//2),:,:])
        self.logger.experiment.add_image('input_masks', mask_grid, self.current_epoch)
        
        
    def set_augmentations(self, augmentation_dict_file):
        self.augmentation_dict = json.load(open(augmentation_dict_file))
        
        
    @staticmethod
    def add_model_specific_args(parent_parser): 
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--in_channels', default=1, type=int)
        parser.add_argument('--out_channels', default=4, type=int)
        parser.add_argument('--feat_channels', default=16, type=int)
        parser.add_argument('--patch_size', default=(64,128,128), type=int, nargs='+')
        parser.add_argument('--out_activation', default='tanh', type=str)
        parser.add_argument('--norm_method', default='instance', type=str)

        # data
        parser.add_argument('--data_norm', default='percentile', type=str)
        parser.add_argument('--data_root', default=r'D:\LfB\pytorchRepo\data\PNAS', type=str) 
        parser.add_argument('--train_list', default=r'D:\LfB\pytorchRepo\data\PNAS_split1_all_train.csv', type=str)
        parser.add_argument('--test_list', default=r'D:\LfB\pytorchRepo\data\PNAS_split1_all_test.csv', type=str)
        parser.add_argument('--val_list', default=r'D:\LfB\pytorchRepo\data\PNAS_split1_all_val.csv', type=str)
        parser.add_argument('--image_groups', default=('data/image',), type=str, nargs='+')
        parser.add_argument('--mask_groups', default=('data/distance', 'data/flow_x', 'data/flow_y', 'data/flow_z'), type=str, nargs='+')
        parser.add_argument('--dist_handling', default='bool_inv', type=str)

        # training params (opt)
        parser.add_argument('--samples_per_epoch', default=-1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--background_weight', default=1, type=float)
        parser.add_argument('--flow_weight', default=1, type=float)
        
        return parser