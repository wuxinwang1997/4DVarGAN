import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(".")

def load_constant(path):
    constant = np.load(os.path.join(path, "constant.npz"))
    constant_mean = np.load(os.path.join(path, "constant_mean.npz"))
    constant_std = np.load(os.path.join(path, "constant_std.npz"))
    vars = ["land_sea_mask", "orography", "latitude", "longitude"]
    out_constant = {}
    for f in vars:
        out_constant[f] = (constant[f] - constant_mean[f]) / constant_std[f]
    out_constant = np.concatenate([out_constant[f] for f in vars], axis=1).astype(np.float32)

    return out_constant

class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out

if __name__ == "__main__":
    load_constant("../../data/train_pred")