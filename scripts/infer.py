#!/usr/bin/env python3

# ----------------------------------------------------
#
# Script to run inference given a new data and a config file
#
# The input file should be an npy file with three columns, being the xyz coordinates
# 
# The file can be specified as a command line argument as an absolute path (--data)
# Or the filename can be given as 'file_name' in the config file. The file should then
# be in a test folder specified by 'test_dir' in the config file within the dataset folder
# specified by 'dataset_path' in the config file
# 
# The output will be writen to a new 'prediction' folder within the dataset folder as a txt file
#
# Usage: type the following in a linux terminal: python3 scripts/inference.py cfg/<config_file_inference.yml> 
#
#
# ----------------------------------------------------

import os
import time
import numpy as np
import torch
import torch.nn as nn
# import open3d.ml as _ml3d
# import open3d.ml.torch as ml3d
import ml3d as _ml3d
import ml3d.torch as ml3d
from open3d.ml.torch.datasets import Custom3D
from pclbox.models import CustomRandLANet, CustomPointTransformer
import argparse


FILE_TYPE = ['npy']


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help='path to configuration file')
    # parser.add_argument("-d", '--data', default=None, help='absolute path to npy file of tree')
    # parser.add_argument("--batch", action='store_true', help='Run on multiple trees. If specified, a dataset_path should be specified in the config file')
    args = parser.parse_args()
    return args


def read_cloud(path):
    if path[-3:] == 'npy':
        pcl = np.load(path)
    elif path[-3:] == 'txt':
        pcl = np.loadtxt(path)
    else:
        raise ValueError('input file should be txt or npy')
    
    data = {
        'point': pcl[:, :3],
        'feat': None,
        'label': np.zeros((pcl.shape[0]), dtype=np.int32)
    }
    return data


if __name__ == '__main__':

    # Load config file
    args = get_arguments()
    cfg_path = args.cfg
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)

    # Get data path, test folder and prediction folder from config file
    data_path = cfg.dataset.get('dataset_path')
    test_dir = cfg.dataset.get('test_dir')
    test_pred_dir = cfg.dataset.get('test_result_folder', 'test_pred')


    # Make prediction directory if it doesn't exist
    if not os.path.exists(os.path.join(data_path, test_pred_dir)):
        os.mkdir(os.path.join(data_path, test_pred_dir))

    # Define model
    if cfg.model.name == 'RandLANet':
        model = CustomRandLANet(**cfg.model) 
    elif cfg.model.name == 'PointTransformer':
        model = CustomPointTransformer(**cfg.model)
    
    # Define pipeline
    pipeline = ml3d.pipelines.SemanticSegmentation(model, **cfg.pipeline)

    # Load models weights
    ckpt_path = cfg.model.ckpt_path
    pipeline.load_ckpt(ckpt_path)

    # Load data if given
    # if args.data is not None :
    #     # If specified as command line argument
    #     data = read_cloud(args.data)
    #     file_name = os.path.dirname(args.data)
    # elif cfg.dataset.get('file_name') is not None:
    #     # If specified in config file
    #     file_name = cfg.dataset.get('file_name')
    #     data = read_cloud(os.path.join(data_path, test_dir, file_name)) 
    # else:
    #     raise ValueError('No input data was specified.')

    # Store filenames of test point clouds in list
    if cfg.dataset.get('file_name') is not None:
        filenames = [cfg.dataset.get('file_name')]
    else:
        filenames = [f for f in os.listdir(os.path.join(data_path, test_dir)) if f[-3:] in FILE_TYPE]
    
    start = time.time()

    for filename in filenames:
        # Read point cloud
        data = read_cloud(os.path.join(data_path, test_dir, filename))

        # Run inference
        pred = pipeline.run_inference(data)
        
        # Save input point cloud with prediction as txt file in 'prediction' folder
        pcl_pred = np.hstack((data['point'], pred['predict_labels'].reshape(-1, 1)))
        
        path_out = os.path.join(data_path, test_pred_dir, filename[:-3] + 'txt')
        np.savetxt(path_out, pcl_pred, fmt='%.3f')
    
    end = time.time()
    print("Inference took:", end-start, "seconds") 