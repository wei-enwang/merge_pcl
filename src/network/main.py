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


image_dir = "../data/imgs/"
latent_dir = "../data/latents/"
stats_dir = "./results/"

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"   # use gpu whenever you can!

# hyperparameters
train_test_split = 0.8
learning_rate = 1e-4
batch_size = 32
seq_len = 10
epochs = 20


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

latent_model = models.resnet_4d()


data_paths = []
latent_paths = []
for data_path in glob.glob(image_dir + '/*'):
    data_paths.append(glob.glob(data_path + '/*'))
for latent_path in glob.glob(latent_dir + '/*'):
    latent_paths.append(glob.glob(latent_path + '/*'))
    
# DONT shuffle the paths because the labels and x are stored in different directories
# Randomly split images in to training and testing datasets. 
# Can also use cross_validate_scheme
n = len(data_paths)
mask = np.random.permutation(n)
train_image_paths = data_paths[mask[:int(train_test_split*n)]]
test_image_paths = data_paths[mask[int(train_test_split*n):]]

train_latent_paths = latent_paths[mask[:int(train_test_split*n)]]
test_latent_paths = latent_paths[mask[int(train_test_split*n):]]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406, None], std=[0.229, 0.224, 0.225, None]),
])

training_data = rgbd_data(train_image_paths, train_latent_paths, transform=preprocess)
testing_data = rgbd_data(test_image_paths, test_latent_paths, transform=preprocess)

# transform images into batches of sequences
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=8, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, num_workers=8, shuffle=False)

model = models.ShapeEncoder().to(device)
loss_function = models.crossMLEloss().to(device)
optim = Adam(model.parameters(), lr=learning_rate)


train_loss, test_loss = train_full_test_once(train_dataloader, test_dataloader, model, loss_function,
                    optim,
                    device,
                    epochs=epochs,
                    vis=True,
                    img_dir=stats_dir)

torch.save(model.state_dict(), stats_dir+"base_randlat.pt")