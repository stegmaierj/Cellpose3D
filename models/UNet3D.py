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

import torch
import torch.nn as nn



class UNet3D_module(nn.Module):
    """Implementation of the 3D U-Net architecture.
    """

    def __init__(self, patch_size, in_channels, out_channels, feat_channels=16, activation='leakyrelu', out_activation='sigmoid', norm_method='none', **kwargs):
        super(UNet3D_module, self).__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.out_activation = out_activation # relu | sigmoid | tanh | hardtanh | none
        self.norm_method = norm_method # instance | batch | none
        
        if self.norm_method == 'instance':
            self.norm = nn.InstanceNorm3d
        elif self.norm_method == 'batch':
            self.norm = nn.BatchNorm3d
        elif self.norm_method == 'none':
            self.norm = nn.Identity
        else:
            raise ValueError('Unknown normalization method "{0}". Choose from "instance|batch|none".'.format(self.norm_method))
        
        if self.norm_method == 'instance':
            self.norm = nn.InstanceNorm3d
        elif self.norm_method == 'batch':
            self.norm = nn.BatchNorm3d
        elif self.norm_method == 'none':
            self.norm = nn.Identity
        else:
            raise ValueError('Unknown normalization method "{0}". Choose from "instance|batch|none".'.format(self.norm_method))
        
        
        # Define layer instances       
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels//2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d1 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.c2 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels*2, kernel_size=3, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d2 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=4, stride=2, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.c3 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*4, kernel_size=3, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d3 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=4, stride=2, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.c4 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*8, kernel_size=3, padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u1 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*8, feat_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c5 = nn.Sequential(
            nn.Conv3d(feat_channels*12, feat_channels*4, kernel_size=3, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u2 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*4, feat_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c6 = nn.Sequential(
            nn.Conv3d(feat_channels*6, feat_channels*2, kernel_size=3, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u3 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*2, feat_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c7 = nn.Sequential(
            nn.Conv3d(feat_channels*3, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.out = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, out_channels, kernel_size=1)
            )
       
        if self.out_activation == 'relu':
            self.out_fcn = nn.ReLU()
        elif self.out_activation == 'sigmoid':
            self.out_fcn = nn.Sigmoid()
        elif self.out_activation == 'tanh':
            self.out_fcn = nn.Tanh()
        elif self.out_activation == 'hardtanh':
            self.out_fcn = nn.Hardtanh(0, 1)
        elif self.out_activation == 'none':
            self.out_fcn = None
        else:
            raise ValueError('Unknown output activation "{0}". Choose from "relu|sigmoid|tanh|hardtanh|none".'.format(self.out_activation))
        

    def forward(self, img):
        
        c1 = self.c1(img)
        d1 = self.d1(c1)
        
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        
        c4 = self.c4(d3)
        
        u1 = self.u1(c4)
        c5 = self.c5(torch.cat((u1,c3),1))
        
        u2 = self.u2(c5)
        c6 = self.c6(torch.cat((u2,c2),1))
        
        u3 = self.u3(c6)
        c7 = self.c7(torch.cat((u3,c1),1))
        
        out = self.out(c7)
        if not self.out_fcn is None:
            out = self.out_fcn(out)
        
        return out
        
    
    
    