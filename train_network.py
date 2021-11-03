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

from argparse import ArgumentParser

import numpy as np
import torch
import glob
import os

from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):

    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    os.makedirs(hparams.output_path, exist_ok=True)
    
    # Load pretrained weights if available
    if not hparams.pretrained is None:
        model.load_pretrained(hparams.pretrained)

    # Resume from checkpoint if available    
    resume_ckpt = None
    if hparams.resume:
        checkpoints = glob.glob(os.path.join(hparams.output_path,'*.ckpt'))
        checkpoints.sort(key=os.path.getmtime)
        if len(checkpoints)>0:
            resume_ckpt = checkpoints[-1]
            print('Resuming from checkpoint: {0}'.format(resume_ckpt))
            
    # Set the augmentations if available
    if not hparams.augmentations is None:
        model.set_augmentations(hparams.augmentations)
        
    # Save a few samples for sanity checks
    print('Saving 20 data samples for sanity checks...')
    model.train_dataloader().dataset.test(os.path.join(hparams.output_path, 'samples'), num_files=20)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    
    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.output_path,
        save_top_k=1,
        monitor='epoch',
        mode='max',
        verbose=True,
        period=5
    )
    
    logger = TestTubeLogger(
        save_dir=hparams.log_path,
        name='lightning_logs_'+hparams.model.lower()
    )
    
    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=False,
        min_nb_epochs=hparams.epochs,
        max_nb_epochs=hparams.epochs,
        early_stop_callback=False,
        resume_from_checkpoint=resume_ckpt
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)



if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'D:\LfB\pytorchRepo\results\Cellpose3D_experiment',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--log_path',
        type=str,
        default=r'D:\LfB\pytorchRepo\logs',
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
        '--no_resume',
        dest='resume',
        action='store_false',
        default=True,
        help='resume training from latest checkpoint'
    )
    
    parent_parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        nargs='+',
        help='path to pretrained model weights'
    )
    
    parent_parser.add_argument(
        '--augmentations',
        type=str,
        default=None,
        help='path to augmentation dict file'
    )
    
    parent_parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='number of epochs'
    )
    
    parent_parser.add_argument(
        '--model',
        type=str,
        default='Cellpose3D',
        help='which model to load (Cellpose3D)'
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
