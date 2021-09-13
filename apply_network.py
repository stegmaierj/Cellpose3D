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

import os
import numpy as np
import torch
import csv
from skimage import io
from argparse import ArgumentParser
from torch.autograd import Variable

from dataloader.h5_dataloader import MeristemH5Tiler as Tiler
from utils.utils import print_timestamp

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    
    
    """
    Main testing routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 0 SANITY CHECKS
    # ------------------------
    if not isinstance(hparams.overlap, (tuple, list)):
        hparams.overlap = (hparams.overlap,) * len(hparams.patch_size)
    if not isinstance(hparams.crop, (tuple, list)):
        hparams.crop = (hparams.crop,) * len(hparams.patch_size)
    assert all([p-2*o-2*c>0 for p,o,c in zip(hparams.patch_size, hparams.overlap, hparams.crop)]), 'Invalid combination of patch size, overlap and crop size.'
    
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    model = model.load_from_checkpoint(hparams.ckpt_path)
    model = model.cuda()
    
    # ------------------------
    # 2 INIT DATA TILER
    # ------------------------
    tiler = Tiler(hparams.test_list, no_mask=hparams.input_batch=='image', no_img=hparams.input_batch=='mask', **vars(hparams))
    fading_map = tiler.get_fading_map()
    fading_map = np.repeat(fading_map[np.newaxis,...], hparams.out_channels, axis=0)
    
    # ------------------------
    # 3 FILE AND FOLDER CHECKS
    # ------------------------
    os.makedirs(hparams.output_path, exist_ok=True)
    file_checklist = []
    if os.path.isfile(os.path.join(hparams.output_path, 'tmp_file_checklist.csv')):
        with open(os.path.join(hparams.output_path, 'tmp_file_checklist.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not len(row)==0:
                    file_checklist.append(row[0])
    
    # ------------------------
    # 4 PROCESS EACH IMAGE
    # ------------------------
    if hparams.num_files is None or hparams.num_files < 0:
        hparams.num_files = len(tiler.data_list)
    else:
        hparams.num_files = np.minimum(len(tiler.data_list), hparams.num_files)
        
    for image_idx in range(hparams.num_files):
        
        # Check if current file has already been processed
        if not any([f==tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1] for f in file_checklist]):
    
            print_timestamp('_'*20)
            print_timestamp('Processing file {0}', [tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1]])
        
            # Initialize current file        
            tiler.set_data_idx(image_idx)
            
            # Determine if the patch size exceeds the image size
            working_size = tuple(np.max(np.array(tiler.locations), axis=0) - np.min(np.array(tiler.locations), axis=0) + np.array(hparams.patch_size))
                
            # Initialize maps (random to overcome memory leaks)        
            predicted_img = np.full((hparams.out_channels,)+working_size, 0, dtype=np.float32)        
            norm_map = np.full((hparams.out_channels,)+working_size, 0, dtype=np.float32)
                    
            for patch_idx in range(tiler.__len__()):
                
                print_timestamp('Processing patch {0}/{1}...',(patch_idx+1, tiler.__len__()))
                
                # Get the mask
                sample = tiler.__getitem__(patch_idx)
                data = Variable(torch.from_numpy(sample[hparams.input_batch][np.newaxis,...]).cuda())
                data = data.float()
                
                # Predict the image
                pred_patch = model(data)
                pred_patch = pred_patch.cpu().data.numpy()
                pred_patch = np.squeeze(pred_patch)
                pred_patch = np.clip(pred_patch, hparams.clip[0], hparams.clip[1])
                            
                # Get the current slice position
                slicing = tuple(map(slice, (0,)+tuple(tiler.patch_start+tiler.global_crop_before), (hparams.out_channels,)+tuple(tiler.patch_end+tiler.global_crop_before)))
                            
                # Add predicted patch and fading weights to the corresponding maps
                predicted_img[slicing] = predicted_img[slicing]+pred_patch*fading_map
                norm_map[slicing] = norm_map[slicing]+fading_map
                
            # Normalize the predicted image
            norm_map = np.clip(norm_map, 1e-5, np.inf)
            predicted_img = predicted_img / norm_map          
            
            # Crop the predicted image to its original size
            slicing = tuple(map(slice, (0,)+tuple(tiler.global_crop_before), (hparams.out_channels,)+tuple(tiler.global_crop_after)))
            predicted_img = predicted_img[slicing]
            
            # Save the predicted image
            predicted_img = np.transpose(predicted_img, (1,2,3,0))
            predicted_img = predicted_img.astype(np.float32)
            if hparams.out_channels > 3:       
                for channel in range(hparams.out_channels):
                    io.imsave(os.path.join(hparams.output_path, 'pred'+str(channel)+'_'+os.path.split(tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1])[-1][:-3]+'.tif'), predicted_img[...,channel])
            else:
                if hparams.out_channels > 1:
                    predicted_img = predicted_img.astype(np.float16)      
                io.imsave(os.path.join(hparams.output_path, 'pred_'+os.path.split(tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1])[-1][:-3]+'.tif'), predicted_img)
    
            # Mark current file as processed
            file_checklist.append(tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1])
            with open(os.path.join(hparams.output_path, 'tmp_file_checklist.csv'), 'w') as f:
                writer = csv.writer(f, delimiter=';')
                for check_file in file_checklist:
                    writer.writerow([check_file])
                
        else:
            print_timestamp('_'*20)
            print_timestamp('Skipping file {0}', [tiler.data_list[image_idx][0 if hparams.input_batch=='image' else 1]])
            
    # Delete temporary checklist if everything has been processed
    os.remove(os.path.join(hparams.output_path, 'tmp_file_checklist.csv'))


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'/path/for/outputs',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--ckpt_path',
        type=str,
        default=r'/path/to/model/epoch=999.ckpt',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of GPUs to use'
    )
    
    parent_parser.add_argument(
        '--overlap',
        type=int,
        default=(8,16,16),
        help='overlap of adjacent patches',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--crop',
        type=int,
        default=(8,32,32),
        help='safety crop of patches',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--input_batch',
        type=str,
        default='image',
        help='which part of the batch is used as input (image | mask)'
    )
    
    parent_parser.add_argument(
        '--model',
        type=str,
        default='Cellpose3D',
        help='Which model to load (Cellpose3D)'
    )
    
    parent_parser.add_argument(
        '--clip',
        type=float,
        default=(-1.0, 1.0),
        help='clipping values for network outputs',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--num_files',
        type=int,
        default=None,
        help='number of files to process'
    )
    
        
    parent_args = parent_parser.parse_known_args()[0]
    
    # load the desired network architecture
    if parent_args.model.lower() == 'cellpose3d':        
        from models.UNet3D_cellpose import UNet3D_cellpose as network
    else:
        raise ValueError('Model {0} unknown.'.format(parent_args.model))
        
    # each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
