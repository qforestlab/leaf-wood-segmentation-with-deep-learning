
import torch
import torch.nn as nn
import numpy as np
import ml3d.torch as ml3d
from ml3d.torch.modules import losses


__all__ = ['CustomRandLANet']


class CustomRandLANet(ml3d.models.RandLANet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_per_class = self.cfg.get('class_weights', None)
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