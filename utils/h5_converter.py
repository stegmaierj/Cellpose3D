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
import h5py
import glob
import numpy as np

from skimage import io, morphology, measure, filters
from scipy.ndimage import distance_transform_edt, zoom, generic_filter
from scipy.spatial import ConvexHull, Delaunay

from utils.utils import print_timestamp



def h5_writer(data_list, save_path, group_root='data', group_names=['image']):
    
    save_path = os.path.abspath(save_path)
    
    assert(len(data_list)==len(group_names)), 'Each data matrix needs a group name'
    
    with h5py.File(save_path, 'w') as f_handle:
        grp = f_handle.create_group(group_root)
        for data, group_name in zip(data_list, group_names):
            grp.create_dataset(group_name, data=data, chunks=True, compression='gzip')
            
            
            
def h5_reader(file, group_identifier='image'):
    
    with h5py.File(file, 'r') as file_handle:
       data = file_handle[group_identifier][:]
                            
    return data



def h52tif(file_dir='', group_names=['data/image']):
    
    # Get all files within the given directory
    filelist = glob.glob(os.path.join(file_dir, '*.h5'))
    
    # Create saving folders
    for group in group_names:
        os.makedirs(os.path.join(file_dir, ''.join(s for s in group if s.isalnum())), exist_ok=True)
    
    # Save each desired group
    for num_file,file in enumerate(filelist):
        print_timestamp('Processing file {0}/{1}', (num_file+1, len(filelist)))
        with h5py.File(file, 'r') as file_handle:
            for group in group_names:
                data = file_handle[group][:]
                io.imsave(os.path.join(file_dir, ''.join(s for s in group if s.isalnum()), os.path.split(file)[-1][:-2]+'tif'), data)
                


def replace_h5_group(source_list, target_list, source_group='data/image', target_group=None):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    if target_group is None: target_group=source_group

    for num_pair, pair in enumerate(zip(source_list, target_list)):
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(target_list)])
        
        # Load the source mask
        with h5py.File(pair[0], 'r') as source_handle:
            source_data = source_handle[source_group][...]
            
        # Save the data to the target file
        with h5py.File(pair[1], 'r+') as target_handle:
            target_data = target_handle[target_group]
            target_data[...] = source_data


            
def add_h5_group(source_list, target_list, source_group='data/distance', target_group=None):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    if target_group is None: target_group=source_group
    
    for num_pair, pair in enumerate(zip(source_list, target_list)):
        
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(source_list)])
            
        # Get the data from the source file
        with h5py.File(pair[0], 'r') as source_handle:
            source_data = source_handle[source_group][...]
            
        # Save the data to the target file
        with h5py.File(pair[1], 'a') as target_handle:
            target_handle.create_dataset(target_group, data=source_data, chunks=True, compression='gzip')



def remove_h5_group(file_list, remove_group='data/nuclei'):
    
    for num_file, file in enumerate(file_list):
        
        print_timestamp('Processing file {0}/{1}...', [num_file+1, len(file_list)])
        
        with h5py.File(file, 'a') as file_handle:
            del file_handle[remove_group]

            
            
def flood_fill_hull(image):    
    
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    
    return out_img



def calculate_flows(instance_mask, bg_label=0):
    
    flow_x = np.zeros(instance_mask.shape, dtype=np.float32)
    flow_y = np.zeros(instance_mask.shape, dtype=np.float32)
    flow_z = np.zeros(instance_mask.shape, dtype=np.float32)
    regions = measure.regionprops(instance_mask)
    for props in regions:
        
        if props.label == bg_label:
            continue
        
        c = props.centroid
        coords = np.where(instance_mask==props.label)
        
        flow_x[coords] = np.tanh((coords[0]-c[0])/5)
        flow_y[coords] = np.tanh((coords[1]-c[1])/5)
        flow_z[coords] = np.tanh((coords[2]-c[2])/5)
    
    return flow_x, flow_y, flow_z
        
        

def rescale_data(data, zoom_factor, order=0):
    
    if any([zf!=1 for zf in zoom_factor]):
        data_shape = data.shape
        data = zoom(data, zoom_factor, order=3)
        print_timestamp('Rescaled image from size {0} to {1}'.format(data_shape, data.shape))
        
    return data



def foreground_from_mip(img):
    
    mip = np.ones_like(img)
    
    for ndim in range(img.ndim):
        mip_tmp = np.max(img, axis=ndim)
        mip_tmp = mip_tmp>0.05
        mip_tmp = np.expand_dims(mip_tmp, ndim)
        mip_tmp = np.repeat(mip_tmp, img.shape[ndim], ndim)
        
        mip *= mip_tmp
        
    return mip
        


def prepare_images(data_path='', folders=[''], identifier='*.tif', descriptor='', normalize=[0,100],\
                   get_distance=False, get_illumination=False, get_variance=False, variance_size=(5,5,5),\
                   fg_selem_size=5, zoom_factor=(1,1,1), channel=0):

    data_path = os.path.abspath(data_path)
     
    for image_folder in folders:
        image_list = glob.glob(os.path.join(data_path, image_folder, identifier))
        for num_file,file in enumerate(image_list):
            
            print_timestamp('Processing image {0}/{1} in "{2}"',(num_file+1, len(image_list), image_folder))
                            
            # load the image
            processed_img = io.imread(file)     
            processed_img = processed_img.astype(np.float32)
            
            # get the desired channel, if the image is a multichannel image
            if processed_img.ndim == 4:
                processed_img = processed_img[...,channel]
            
            # rescale the image
            processed_img = rescale_data(processed_img, zoom_factor, order=3)
            
            # normalize the image
            perc1, perc2 = np.percentile(processed_img, list(normalize))
            processed_img -= perc1
            processed_img /= (perc2-perc1)
            processed_img = np.clip(processed_img, 0, 1)
            processed_img = processed_img.astype(np.float32)
                
            save_imgs = [processed_img,]
            save_groups = ['image',]
            
            if get_illumination:
                
                print_timestamp('Extracting illumination image...')
                
                # create downscales image for computantially intensive processing
                small_img = processed_img[::2,::2,::2]
                
                # create an illuminance image (downscale for faster processing)
                illu_img = morphology.closing(small_img, selem=morphology.ball(7))
                illu_img = filters.gaussian(illu_img, 2).astype(np.float32)
                
                # rescale illuminance image
                illu_img = np.repeat(illu_img, 2, axis=0)
                illu_img = np.repeat(illu_img, 2, axis=1)
                illu_img = np.repeat(illu_img, 2, axis=2)
                dim_missmatch = np.array(processed_img.shape)-np.array(illu_img.shape)
                if dim_missmatch[0]<0: illu_img = illu_img[:dim_missmatch[0],...]
                if dim_missmatch[1]<0: illu_img = illu_img[:,:dim_missmatch[1],:]
                if dim_missmatch[2]<0: illu_img = illu_img[...,:dim_missmatch[2]]
                
                save_imgs.append(illu_img.astype(np.float32))
                save_groups.append('illumination')
                
            if get_distance:
                
                print_timestamp('Extracting distance image...')
                
                # create downscales image for computantially intensive processing
                small_img = processed_img[::4,::4,::4]
                
                # find suitable threshold
                thresh = filters.threshold_otsu(small_img)
                fg_img = small_img > thresh
                
                # remove noise and fill holes
                fg_img = morphology.binary_closing(fg_img, selem=morphology.ball(fg_selem_size))
                fg_img = morphology.binary_opening(fg_img, selem=morphology.ball(fg_selem_size))
                fg_img = flood_fill_hull(fg_img)
                fg_img = fg_img.astype(np.bool)
                
                # create distance transform
                fg_img = distance_transform_edt(fg_img) - distance_transform_edt(~fg_img)
                
                # rescale distance image
                fg_img = np.repeat(fg_img, 4, axis=0)
                fg_img = np.repeat(fg_img, 4, axis=1)
                fg_img = np.repeat(fg_img, 4, axis=2)
                dim_missmatch = np.array(processed_img.shape)-np.array(fg_img.shape)
                if dim_missmatch[0]<0: fg_img = fg_img[:dim_missmatch[0],...]
                if dim_missmatch[1]<0: fg_img = fg_img[:,:dim_missmatch[1],:]
                if dim_missmatch[2]<0: fg_img = fg_img[...,:dim_missmatch[2]]
                
                save_imgs.append(fg_img.astype(np.float32))
                save_groups.append('distance')
                
            if get_variance:
                
                print_timestamp('Extracting variance image...')
                
                # create downscales image for computantially intensive processing
                small_img = processed_img[::4,::4,::4]
                
                # create variance image
                std_img = generic_filter(small_img, np.std, size=variance_size)
                
                # rescale variance image
                std_img = np.repeat(std_img, 4, axis=0)
                std_img = np.repeat(std_img, 4, axis=1)
                std_img = np.repeat(std_img, 4, axis=2)
                dim_missmatch = np.array(processed_img.shape)-np.array(std_img.shape)
                if dim_missmatch[0]<0: std_img = std_img[:dim_missmatch[0],...]
                if dim_missmatch[1]<0: std_img = std_img[:,:dim_missmatch[1],:]
                if dim_missmatch[2]<0: std_img = std_img[...,:dim_missmatch[2]]
                
                save_imgs.append(std_img.astype(np.float32))
                save_groups.append('variance')
            
            # save the data
            save_name = os.path.split(file)[-1]
            save_name = os.path.join(data_path, image_folder, descriptor+save_name[:-4]+'.h5')
            h5_writer(save_imgs, save_name, group_root='data', group_names=save_groups)
      
      
      
      
        
def prepare_masks(data_path='', folders=[''], identifier='*.tif', descriptor='',\
                  bg_label=0, get_flows=True, get_boundary=False, get_seeds=False, get_distance=True,\
                  corrupt_prob=0.0, zoom_factor=(1,1,1)):
     
    data_path = os.path.abspath(data_path)
    
    for mask_folder in folders:
        mask_list = glob.glob(os.path.join(data_path, mask_folder, identifier))
        experiment_identifier = 'corrupt'+str(corrupt_prob).replace('.','') if corrupt_prob > 0 else ''
        os.makedirs(os.path.join(data_path, mask_folder, experiment_identifier), exist_ok=True)
        for num_file,file in enumerate(mask_list):
            
            print_timestamp('Processing mask {0}/{1} in {2}', (num_file+1, len(mask_list), mask_folder))
            
            # load the mask
            instance_mask = io.imread(file)
            instance_mask = instance_mask.astype(np.uint16)
            instance_mask[instance_mask==bg_label] = 0
                            
            # rescale the mask
            instance_mask = rescale_data(instance_mask, zoom_factor, order=0)
            
            if corrupt_prob > 0:
                # Randomly merge neighbouring instances
                labels = list(set(np.unique(instance_mask))-set([bg_label]))
                instance_mask_eroded = morphology.erosion(instance_mask, selem=morphology.ball(3))
                instance_mask_dilated = morphology.dilation(instance_mask, selem=morphology.ball(3))
                for label in labels:
                    if np.random.rand() < corrupt_prob:
                        neighbour_labels = list(instance_mask_eroded[instance_mask==label]) + list(instance_mask_dilated[instance_mask==label])
                        neighbour_labels = list(set(neighbour_labels)-set([label,]))
                        if len(neighbour_labels) > 0:
                            replace_label = np.random.choice(neighbour_labels)                            
                            instance_mask[instance_mask==label] = replace_label
                            
            save_groups = ['instance',]
            save_masks = [instance_mask,] 
            
            # get the boundary mask
            if get_boundary:
                membrane_mask = morphology.dilation(instance_mask, selem=morphology.ball(2)) - instance_mask
                membrane_mask = membrane_mask != 0
                membrane_mask = membrane_mask.astype(np.float32)
                save_groups.append('boundary')
                save_masks.append(membrane_mask)
            
            # get the distance mask
            if get_distance:
                fg_img = instance_mask[::4,::4,::4]
                fg_img = flood_fill_hull(fg_img>0)
                fg_img = fg_img.astype(np.bool)
                distance_mask = distance_transform_edt(fg_img) - distance_transform_edt(~fg_img)
                distance_mask = distance_mask.astype(np.float32)
                distance_mask = np.repeat(distance_mask, 4, axis=0)
                distance_mask = np.repeat(distance_mask, 4, axis=1)
                distance_mask = np.repeat(distance_mask, 4, axis=2)
                dim_missmatch = np.array(instance_mask.shape)-np.array(distance_mask.shape)
                if dim_missmatch[0]<0: distance_mask = distance_mask[:dim_missmatch[0],...]
                if dim_missmatch[1]<0: distance_mask = distance_mask[:,:dim_missmatch[1],:]
                if dim_missmatch[2]<0: distance_mask = distance_mask[...,:dim_missmatch[2]]
                save_groups.append('distance')
                save_masks.append(distance_mask)
            
            # get the centroid mask
            if get_seeds:
                centroid_mask = np.zeros(instance_mask.shape, dtype=np.float32)
                regions = measure.regionprops(instance_mask)
                
                for props in regions:
                    
                    if props.label == bg_label:
                        continue
                    
                    c = props.centroid
                    centroid_mask[np.int(c[0]), np.int(c[1]), np.int(c[2])] = 1
                
                save_groups.append('seeds')
                save_masks.append(centroid_mask)
            
            # calculate the flow field
            if get_flows:
                
                flow_x, flow_y, flow_z = calculate_flows(instance_mask, bg_label=bg_label)
                    
                save_groups.extend(['flow_x','flow_y', 'flow_z'])
                save_masks.extend([flow_x, flow_y, flow_z])
                    
            # save the data
            save_name = os.path.split(file)[-1]
            save_name = os.path.join(data_path, mask_folder, experiment_identifier, descriptor+save_name[:-4]+'.h5')
            h5_writer(save_masks, save_name, group_root='data', group_names=save_groups)
