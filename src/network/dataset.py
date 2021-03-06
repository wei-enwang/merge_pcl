import torch
import json

from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class rgbd_data(Dataset):
    """
    Class for pybullet rgbd images. 

    Args:
        data_paths:
            directories of the saved images
        transform (a function or a list of functions):
            transform(s) applied to features (x)

    """

    def __init__(self, data_paths, latent_paths, seq_len=10, max_objects=10, transform=None, idx_mask=None):

        self.data_paths = data_paths
        self.latent_paths = latent_paths
        self.seq_len = seq_len
        self.max_objects = max_objects
        self.transform = transform
        # TODO: replace the following with the latents generated by occupancy network
        with open('../data/latents/trial_latents.json', ) as data:
            self.gt_latents_dict = json.load(data)
        # For our ease of implementation, suppose number of images is divisible by the length of sequences
        assert len(data_paths) % seq_len == 0
        self.idxs = np.arange(len(data_paths)).reshape((-1,seq_len))
        if idx_mask is not None:
            self.idxs = self.idxs[idx_mask,:]

    def __len__(self):
        assert len(self.data_paths) == len(self.latent_paths)
        return len(self.idxs)

    def __getitem__(self, idx):

        img_seq = []
        gt_latents_seq = []
        for i in self.idxs[idx,:]:
            img_path = self.data_paths[i]
            
            img = np.load(img_path).astype(int)
            # Apply transform to image
            img = Image.fromarray(img, mode="RGBA")
            img = self.transform(img)
            img_seq.append(img)
        
            latents = self.latent_paths[i]  
            gt_latents = np.load(latents) 
            gt_latents_seq.append(gt_latents)

        img_seq = torch.from_numpy(np.stack(img_seq, axis=0).astype(float))
        gt_latents_seq = torch.from_numpy(np.stack(gt_latents_seq, axis=0))
        return img_seq, gt_latents_seq

