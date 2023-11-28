#!/usr/bin/env python3

# ----------------------------------------------------
#
# Script to train a 3D deep learning model based on a
# config file, using the open3d-ml framework.
#
# Usage: type the following in a linux terminal: python3 scripts/train.py cfg/<config_file.yml> 
#
# Run 'tensorboard --logdir=train_log' to visualise training in tensorboard
#
# ----------------------------------------------------

import time
import numpy as np
import torch
import torch.nn as nn
# import open3d.ml as _ml3d
# import open3d.ml.torch as ml3d
import ml3d as _ml3d
import ml3d.torch as ml3d
from open3d.ml.torch.datasets import Custom3D
from open3d.ml.torch.modules import losses
from pclbox.models import CustomRandLANet, CustomPointTransformer, CustomKPConv
import argparse

import logging
logging.basicConfig(level=logging.DEBUG)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help='path to configuration file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # Load config file
    args = get_arguments()
    cfg_path = args.cfg
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)

    # Define dataset
    dataset = Custom3D(**cfg.dataset)

    # Define model
    if cfg.model.name == 'RandLANet':
        model = CustomRandLANet(**cfg.model) # ml3d.models.RandLANet(**cfg.model)
    elif cfg.model.name == 'PointTransformer':
        model = CustomPointTransformer(**cfg.model)
    elif cfg.model.name == 'KPConv':
        model = CustomKPConv(**cfg.model)

    # Define pipeline
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset, **cfg.pipeline)

    # Train model
    start = time.time()
    pipeline.run_train()
    end = time.time()
    print("Training took:", end-start, "seconds") 
