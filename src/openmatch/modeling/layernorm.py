import logging
import os
import json

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(
            self,
            normalized_shape,
            eps: float = 1e-5,
            elementwise_affine: bool = True,
    ):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.config = {'normalized_shape': normalized_shape, 'eps': eps, 'elementwise_affine': elementwise_affine}

    def forward(self, rep: Tensor = None):
        return self.layernorm(rep)

    @classmethod
    def load(cls, ckpt_dir: str):
        logger.info(f'Loading layernorm from {ckpt_dir}')
        model_path = os.path.join(ckpt_dir, 'layernorm.pt')
        config_path = os.path.join(ckpt_dir, 'layernorm_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(model_path))
        return model

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'layernorm.pt'))
        with open(os.path.join(save_path, 'layernorm_config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)