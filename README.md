# Cellpose3D

This repository contains code used for the 3D Cellpose extension proposed in [Preprint](https://arxiv.org/abs/2105.00794).

If you use this code, please cite:<br>
<b>D. Eschweiler, J. Stegmaier, 
"Robust 3D Cell Segmentation: Extending the View of Cellpose", arXiv:2105.00794, 2021.</b>


### Data Preparation
The data needs to be in a hdf5 format containing image data for the network input and gradient information as output.
The data is assumed to be in a structure similar to the following schematic.

`-|data_root`<br>
`----|experiment1`<br>
`--------|images_as_tif`<br>
`--------|masks_as_tif`<br>
`----|experiment2`<br>
`--------|images_as_tif`<br>
`--------|masks_as_tif`<br>

To prepare your own data, proceed as explained in the following steps:
1. Convert the data using `utils.h5_converter.prepare_images` and `utils.h5_converter.prepare_masks` to prepare image and mask data, respectively.
2. Create a .csv filelist using `utils.csv_generator.create_csv`, while the input is assumed to be a list of tuples containing image-mask pairs -> <br>
`[('experiment1/images_converted/im_1.h5', 'experiment1/masks_converted/mask_1.h5'),`<br>
  `...,`<br>
  `('experiment2/images_converted/im_n.h5', 'experiment2/masks_converted/mask_n.h5')]`<br>
  
  
### Training and Application
For training and application use the provided scripts and make sure to adjust the data paths in the `models.UNet3D_cellpose` accordingly.