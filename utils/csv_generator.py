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
import glob
import csv
import numpy as np


def get_files(folders, data_root='', descriptor='', filetype='tif'):
    
    filelist = []
    
    for folder in folders:
        files = glob.glob(os.path.join(data_root, folder, '*'+descriptor+'*.'+filetype))
        filelist.extend([os.path.join(folder, os.path.split(f)[-1]) for f in files])
        
    return filelist
        
    
        
def read_csv(list_path, data_root=''):
    
    filelist = []    
    with open(list_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row)==0 or np.sum([len(r) for r in row])==0: continue
            row = [os.path.join(data_root, r) for r in row]
            filelist.append(row)
    return filelist
   
    
        
def create_csv(data_list, save_path='list_folder/experiment_name', test_split=0.2, val_split=0.1, shuffle=False):
        
    if shuffle:
        np.random.shuffle(data_list)
    
    # Get number of files for each split
    num_files = len(data_list)
    num_test_files = int(test_split*num_files)
    num_val_files = int((num_files-num_test_files)*val_split)
    num_train_files = num_files - num_test_files - num_val_files
    
    # Get file indices
    file_idx = np.arange(num_files)
    
    # Save csv files
    if num_test_files > 0:
        test_idx = sorted(np.random.choice(file_idx, size=num_test_files, replace=False))
        with open(save_path+'_test.csv', 'w') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in test_idx:
                writer.writerow(data_list[idx])
    else:
        test_idx = []
        
    if num_val_files > 0:
        val_idx = sorted(np.random.choice(list(set(file_idx)-set(test_idx)), size=num_val_files, replace=False))
        with open(save_path+'_val.csv', 'w') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in val_idx:
                writer.writerow(data_list[idx])
    else:
        val_idx = []
    
    if num_train_files > 0:
        train_idx = sorted(list(set(file_idx) - set(test_idx) - set(val_idx)))
        with open(save_path+'_train.csv', 'w') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in train_idx:
                writer.writerow(data_list[idx])
            
            