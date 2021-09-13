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

from utils.postprocessing import apply_cellpose



def main(hparams):
    
    apply_cellpose(hparams.filedir, fg_identifier=hparams.fg_identifier, x_identifier=hparams.flowx_identifier,\
                   y_identifier=hparams.flowy_identifier, z_identifier=hparams.flowz_identifier,\
                   min_diameter=hparams.min_diam, max_diameter=hparams.max_diam, step_size=hparams.step_size,\
                   smoothing_var=hparams.smoothing_var, niter=hparams.niter, njobs=hparams.njobs,\
                   fg_thresh=hparams.fg_thresh, fg_overlap_thresh=hparams.fg_overlap_thresh,\
                   flow_thresh=hparams.flow_thresh, convexity_thresh=hparams.convexity_thresh,\
                   normalize_flows=hparams.normalize_flows, invert_flows=hparams.invert_flows)
   

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--filedir',
        type=str,
        default=r'D:\LfB\pytorchRepo\results\Cellpose_3D',
        help='Directory of the predictions'
    )
    
    parent_parser.add_argument(
        '--fg_identifier',
        type=str,
        default='pred0',
        help='Foreground prediction identifier'
    )
    
    parent_parser.add_argument(
        '--flowx_identifier',
        type=str,
        default='pred1',
        help='x-flow prediction identifier'
    )
    
    parent_parser.add_argument(
        '--flowy_identifier',
        type=str,
        default='pred2',
        help='y-flow prediction identifier'
    )
    
    parent_parser.add_argument(
        '--flowz_identifier',
        type=str,
        default='pred3',
        help='z-flow prediction identifier'
    )
    
    parent_parser.add_argument(
        '--min_diam',
        type=int,
        default=5,
        help='Maximum cell diameter'
    )
    
    parent_parser.add_argument(
        '--max_diam',
        type=int,
        default=100,
        help='Minimum cell diameter'
    )
    
    parent_parser.add_argument(
        '--step_size',
        type=int,
        default=1,
        help='Iteration step size'
    )
    
    parent_parser.add_argument(
        '--smoothing_var',
        type=int,
        default=1,
        help='Variance of Gaussian flow field smoothing'
    )
    
    parent_parser.add_argument(
        '--niter',
        type=int,
        default=100,
        help='Number of iterations'
    )
    
    parent_parser.add_argument(
        '--njobs',
        type=int,
        default=4,
        help='Number of jobs'
    )
    
    parent_parser.add_argument(
        '--fg_thresh',
        type=float,
        default=0.5,
        help='Foreground threshold'
    )
    
    parent_parser.add_argument(
        '--fg_overlap_thresh',
        type=float,
        default=0.5,
        help='Foreground overlap threshold'
    )
    
    parent_parser.add_argument(
        '--flow_thresh',
        type=float,
        default=0.8,
        help='Flow field threshold'
    )
    
    parent_parser.add_argument(
        '--convexity_thresh',
        type=float,
        default=0.1,
        help='Cell convexity threshold'
    )
    
    parent_parser.add_argument(
        '--normalize_flows',
        dest='normalize_flows',
        action='store_true',
        default=False,
        help='Normalizing the flow fields'
    )
        
    parent_parser.add_argument(
        '--invert_flows',
        dest='invert_flows',
        action='store_true',
        default=False,
        help='Inverting the flow fields'
    )
    
        
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
