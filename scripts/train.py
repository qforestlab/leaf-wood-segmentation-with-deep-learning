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

import numpy as np
import torch
import torch.nn as nn
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.torch.datasets import Custom3D
from open3d.ml.torch.modules import losses

import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help='path to configuration file')
    args = parser.parse_args()
    return args


class CustomRandLANet(ml3d.models.RandLANet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_per_class = cfg.dataset.get('class_weights', None)
        self.class_weights = self.get_class_weights(num_per_class) if num_per_class is not None else None
        self.cce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def get_class_weights(self, num_per_class):
        num_per_class = np.array(num_per_class, dtype=np.float32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return torch.tensor(ce_label_weight)
    
    def get_loss(self, Loss, results, inputs, device):
        labels = inputs['data']['labels']
        scores, labels = losses.filter_valid_label(
            results, 
            labels, 
            self.cfg.num_classes,
            self.cfg.ignored_label_inds,
            device,
        )
        loss = self.cce_loss(scores, labels)
        return loss, labels, scores


if __name__ == '__main__':

    # Load config file
    args = get_arguments()
    cfg_path = args.cfg
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)

    # Instantiate dataset, model and pipeline
    dataset = Custom3D(**cfg.dataset)
    model = CustomRandLANet(**cfg.model) # ml3d.models.RandLANet(**cfg.model)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset, **cfg.pipeline)

    # Train model
    pipeline.run_train()