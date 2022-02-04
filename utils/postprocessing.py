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
import torch
import numpy as np

from skimage import io
from skimage.morphology import label, binary_dilation, binary_closing, ball
from skimage.measure import regionprops
from scipy.ndimage import filters, zoom

from utils.h5_converter import calculate_flows
from utils.utils import print_timestamp



def apply_cellpose(filedir, fg_identifier='pred0', x_identifier='pred1', y_identifier='pred2', z_identifier='pred3',\
                   min_diameter=5, max_diameter=100, step_size=1, smoothing_var=1, niter=100, njobs=4,\
                   fg_thresh=0.5, flow_thresh=0.8, fg_overlap_thresh=0.5, convexity_thresh=0.1,\
                   normalize_flows=True, invert_flows=False, verbose=True, overwrite=False):
    
    filelist = os.listdir(filedir)
    fg_files = sorted([f for f in filelist if fg_identifier in f])
    x_files = sorted([f for f in filelist if x_identifier in f])
    y_files = sorted([f for f in filelist if y_identifier in f])
    z_files = sorted([f for f in filelist if z_identifier in f])
    
    print_timestamp('Predictions for {0} different files found...',[len(fg_files)])
    
    i = 0
    for fg_file, x_file, y_file, z_file in zip(fg_files, x_files, y_files, z_files):
        i += 1
        print_timestamp('Processing file {0}/{1}: {2}...',[i,len(fg_files), fg_file[len(fg_identifier)+1:]])
        
        save_path = os.path.join(filedir,'instances'+fg_file[len(fg_identifier):])
        
        if save_path in filelist and not overwrite:
            print_timestamp('File has already been processed. Skipping {0}/{1}: {2}...',[i,len(fg_files), fg_file[len(fg_identifier)+1:]])
            continue
        
        fg_map = io.imread(os.path.join(filedir, fg_file))
        flow_x = io.imread(os.path.join(filedir, x_file))
        flow_y = io.imread(os.path.join(filedir, y_file))
        flow_z = io.imread(os.path.join(filedir, z_file))
        save_path = os.path.join(filedir,'instances'+fg_file[len(fg_identifier):])
        cellpose_flowcontrol(fg_map, flow_x, flow_y, flow_z, save_path=save_path,\
                             niter=niter, njobs=njobs, min_diameter=min_diameter,\
                             max_diameter=max_diameter, step_size=step_size, smoothing_var=smoothing_var,\
                             fg_thresh=fg_thresh, fg_overlap_thresh=fg_overlap_thresh,\
                             flow_thresh=flow_thresh, convexity_thresh=convexity_thresh,\
                             normalize_flows=normalize_flows, invert_flows=invert_flows, verbose=verbose)
    



def cellpose_flowcontrol(fg_map, flow_x, flow_y, flow_z, niter=100, njobs=4,\
                         min_diameter=10, max_diameter=100, step_size=1, smoothing_var=1, fg_thresh=0.5,\
                         fg_overlap_thresh=0.5, flow_thresh=0.5, save_path=None, normalize_flows=True,\
                         verbose=True, convexity_thresh=0.1, invert_flows=False):
    
    fg_map = fg_map.astype(np.float32)
    flow_x = flow_x.astype(np.float32)
    flow_y = flow_y.astype(np.float32)
    flow_z = flow_z.astype(np.float32)
    
    # Normalize flow maps
    if normalize_flows:
        if verbose: print_timestamp('Normalizing the flow fields...')
        max_flow = np.max(np.abs(np.array((flow_x,flow_y,flow_z))))
        flow_x /= max_flow
        flow_y /= max_flow
        flow_z /= max_flow    
        
    if invert_flows:
        if verbose: print_timestamp('Inverting the flow fields...')
        flow_x *= -1
        flow_y *= -1
        flow_z *= -1        
    
    # Smooth the flow fields
    if smoothing_var>0:
        if verbose: print_timestamp('Smoothing the flow fields...')
        flow_x = filters.gaussian_filter(flow_x, smoothing_var).astype(np.float32)
        flow_y = filters.gaussian_filter(flow_y, smoothing_var).astype(np.float32)
        flow_z = filters.gaussian_filter(flow_z, smoothing_var).astype(np.float32)
        
    # Prepare flow arrays
    flow_x_torch = torch.from_numpy(flow_x*step_size)
    flow_y_torch = torch.from_numpy(flow_y*step_size)
    flow_z_torch = torch.from_numpy(flow_z*step_size)
    
    if verbose: print_timestamp('Initializing positions...')      
    # Initialize each position
    pos_x, pos_y, pos_z = np.indices(fg_map.shape, dtype=np.float32)
    pos_x = torch.from_numpy(pos_x)
    pos_y = torch.from_numpy(pos_y)
    pos_z = torch.from_numpy(pos_z)
        
    # Iteratively move each pixel along the flow field   
    for i in range(niter):
        
        if verbose: print_timestamp('Iteration {0:0{2}}/{1}...', args=[i+1, niter, len(str(niter))])
        
        # Get updated position
        new_x = torch.clamp(pos_x - flow_x_torch[(pos_x.long(), pos_y.long(), pos_z.long())], 0, fg_map.shape[0]-1)
        new_y = torch.clamp(pos_y - flow_y_torch[(pos_x.long(), pos_y.long(), pos_z.long())], 0, fg_map.shape[1]-1)
        new_z = torch.clamp(pos_z - flow_z_torch[(pos_x.long(), pos_y.long(), pos_z.long())], 0, fg_map.shape[2]-1)
        
        # Assign new positions
        pos_x = new_x
        pos_y = new_y
        pos_z = new_z        
                                
    # Convert tensor to numpy
    pos_x = pos_x.data.numpy().astype(np.uint16)
    pos_y = pos_y.data.numpy().astype(np.uint16)
    pos_z = pos_z.data.numpy().astype(np.uint16)
    
    # Create map of final positions
    pos_mask = np.zeros_like(fg_map, dtype=np.bool)
    pos_mask[pos_x,pos_y,pos_z] = True
        
    # Reconstruct each clustered instance
    if verbose: print_timestamp('Reconstructing instances...')
    pos_mask = binary_closing(pos_mask, selem=ball(min_diameter//2))
    pos_mask = label(pos_mask)
    instances = pos_mask[(pos_x, pos_y, pos_z)]
    
    # Remove noise and huge instance clusters
    fg_map = binary_dilation(fg_map>fg_thresh, ball(min_diameter//2))
    if verbose: print_timestamp('Removing oversized instance clusters and noise...')
    labels, counts = np.unique(instances, return_counts=True)
    remove_labels = [l for l,c in zip(labels, counts) if c<4/3*np.pi*(min_diameter/2)**3 or c>4/3*np.pi*(max_diameter/2)**3 or np.sum(fg_map[instances==l])/c<fg_overlap_thresh]    
    if len(remove_labels)>0:
        instances[np.isin(instances, remove_labels)] = 0
    
    # Remove bad flow masks
    if verbose: print_timestamp('Removing instances with bad flow and bad convexity...')
    recon_flow_x, recon_flow_y, recon_flow_z = calculate_flows(instances)
    flow_error = (np.abs(flow_x-recon_flow_x) + np.abs(flow_y-recon_flow_y) + np.abs(flow_z-recon_flow_z))/3
    remove_labels = []
    error_map = np.zeros_like(instances, dtype=np.float32)
    for region in regionprops(instances):
        error_map[instances==region.label] = np.mean(flow_error[instances==region.label])
        if np.mean(flow_error[instances==region.label]) > flow_thresh:
            remove_labels.append(region.label)
        if region.minor_axis_length/region.major_axis_length < convexity_thresh:
            remove_labels.append(region.label)
    if len(remove_labels)>0:
            instances[np.isin(instances, remove_labels)] = 0
        
    # Adjust final label range
    if verbose: print_timestamp('Adjusting final label range...')
    instances = label(instances)
    instances = instances.astype(np.uint32)

    if not save_path is None:
        if verbose: print_timestamp('Saving results...')
        io.imsave(save_path, instances)
    
    if verbose: print_timestamp('Finished!')
        
    return instances
    

    
    
def crop_images(filepaths, start_idx=(0,0,0), end_idx=(200,200,100), size=None, save_path=None):
    
    assert not size is None or not end_idx is None, 'Either an end point or a size must be given.'
    
    if end_idx is None and not size is None:
        end_idx = [start+s for start,s in zip(start_idx, size)]
    
    slicing = tuple(map(slice, start_idx, end_idx))
    
    for file in filepaths:
        
        save_name = os.path.split(file)[-1]
        if save_path is None:
            save_path = os.path.split(file)[0]
        
        data = io.imread(file)
        data = data[slicing]
        
        io.imsave(os.path.join(save_path, '_'.join([str(s) for s in start_idx])+'_'+save_name), data)
        
        
        
        
def scale_images(filepaths, zoom_factor=(1,1,1), zoom_order=3, save_identifier=''):
    
    for num_file, filepath in enumerate(filepaths):
        
        print_timestamp('Processing file {0}/{1}: {2}', [num_file+1, len(filepaths), os.path.split(filepath)[1]])
        
        img = io.imread(filepath)
        img = zoom(img, zoom_factor, order=zoom_order)
        io.imsave(os.path.join(os.path.split(filepath)[0], save_identifier+os.path.split(filepath)[1]), img)
                
        
        
def resize_images(filepaths, size=(100,100,100), zoom_order=3, save_identifier=''):
    
    for num_file, filepath in enumerate(filepaths):
        
        print_timestamp('Processing file {0}/{1}: {2}', [num_file+1, len(filepaths), os.path.split(filepath)[1]])
        
        img = io.imread(filepath)
        zoom_factor = [s/i for s,i in zip(size,img.shape)]
        img = zoom(img, zoom_factor, order=zoom_order)
        io.imsave(os.path.join(os.path.split(filepath)[0], save_identifier+os.path.split(filepath)[1]), img)
        
        
        
def flip_dim(filepaths, axis=0, save_identifier=''):
    
    for num_file, filepath in enumerate(filepaths):
        
        print_timestamp('Processing file {0}/{1}: {2}', [num_file+1, len(filepaths), os.path.split(filepath)[1]])
        
        img = io.imread(filepath)
        img = np.flip(img, axis=axis)
        io.imsave(os.path.join(os.path.split(filepath)[0], save_identifier+os.path.split(filepath)[1]), img)
