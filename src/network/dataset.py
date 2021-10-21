import torch
import json
import glob
from torch.utils.data import Dataset
import numpy as np


class rgbd_data(Dataset):
    """
    Class for pybullet rgbd images. 

    Args:
        data_dir:
            directories of the saved images
        transform (a function or a list of functions):
            transform(s) applied to features (x)
        target_transform (a function or a list of functions):
            transform applied to label (y)
    """

    def __init__(self, data_paths, transform=None, target_transform=None, target_inv_transform=None):

        self.data_paths = data_paths
        self.transform = transform
        self.target_transform = target_transform
        self.target_inv_transform = target_inv_transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        img = np.load(img_path)

        gt_latents = None # TODO: implement latent generator 

        return img, gt_latents

