from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

import random
import torch
import torch.cuda
import torch.nn as nn
import numpy as np
import glob

import models
from dataset import rgbd_data
from train_test import train_full_test_once
import transform


image_dir = "../data/imgs/"
latent_dir = "../data/latents/"
stats_dir = "./results/"

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"   # use gpu whenever you can!

# hyperparameters
train_test_split = 0.8
learning_rate = 1e-4
batch_size = 32
dropout = 0.
seq_len = 10
epochs = 20

# scene settings
num_objects = 10


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

data_paths = sorted(glob.glob(image_dir + '*.npy'))
latent_paths = sorted(glob.glob(latent_dir + '*.npy'))
    
# DONT shuffle the paths because the labels and x are stored in different directories
# Randomly split images in to training and testing datasets. 
# Can also use cross_validate_scheme
n = len(data_paths)//seq_len
mask = np.random.permutation(n)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transform.Rescale(224)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406, None], std=[0.229, 0.224, 0.225, None]),
])

# training and testing data are separated by mask
# print(data_paths)
training_data = rgbd_data(data_paths, latent_paths, 
                          transform=preprocess, idx_mask=mask[:int(train_test_split*n)])
testing_data = rgbd_data(data_paths, latent_paths, 
                         transform=preprocess, idx_mask=mask[int(train_test_split*n):])

# transform images into batches of sequences
# TODO: change num_workers to 8 when running on GPU
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

model = models.ShapeEncoder(num_obj=num_objects, dropout=dropout).to(device)
loss_function = models.crossMSEloss().to(device)
optim = Adam(model.parameters(), lr=learning_rate)


train_loss, test_loss = train_full_test_once(train_dataloader, test_dataloader, model, loss_function,
                    optim,
                    device,
                    epochs=epochs,
                    vis=True,
                    img_dir=stats_dir)

torch.save(model.state_dict(), stats_dir+"base_randlat.pt")
