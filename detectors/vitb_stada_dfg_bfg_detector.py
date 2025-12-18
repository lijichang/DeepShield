import logging
from einops import rearrange

import torch.nn as nn

import sys
# current_file_path = os.path.abspath(__file__)
# parent_dir = os.path.dirname(os.path.dirname(current_file_path))
# project_root_dir = os.path.dirname(parent_dir)
# sys.path.append(parent_dir)
# sys.path.append(project_root_dir)
sys.path.append("..")

from torch import nn
from networks.vitb_stada import clip_vit_base_patch16_adapter24x384_dfg_bfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.cuda.amp import autocast


random_select = True
no_time_pool = True

logger = logging.getLogger(__name__)

class vitbstDetector(nn.Module):
    def __init__(self, clip_size=12, num_clips=4):
        super().__init__()
        self.clip_size = clip_size
        self.num_clips = num_clips
        self.default_cfg = {'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD}

        self.vitb = clip_vit_base_patch16_adapter24x384_dfg_bfg(num_classes=2)

        
    # @autocast(True)
    def forward(self, x, train=False, bfg=False):
        x = rearrange(x, 'b (n t) c h w -> (b n) t c h w', t=self.clip_size, c=3)
        x = x.permute(0, 2, 1, 3, 4)
        pred, feats, patch_pred = self.vitb(x, self.num_clips, train, bfg) #feats[bn,768], patch_pred[bnt*14*14,2]
        
        return pred, feats, patch_pred