import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import utils

def train_loop(dataloader, model, loss_fn, optimizer=None, device="cpu"):
    size = len(dataloader.dataset)
    for X, y in dataloader:
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if optimizer:
            optimizer.step()


    pred = model(X)

    SBP_loss = loss_fn(pred[:,0], y[:,0]).item()
    DBP_loss = loss_fn(pred[:,1], y[:,1]).item()

    MAE = nn.L1Loss()(pred[:,0], y[:,0]).item()
    mean = np.mean((pred - y)[:,0].tolist())
    std = np.std((pred-y)[:,0].tolist())

    print(f"SBP Training loss: {SBP_loss:>5f}")
    print(f"DBP Training loss: {DBP_loss:>5f}")
    
    return np.array([SBP_loss, DBP_loss, MAE, mean, std])


def test_loop(dataloader, model, loss_fn, tar_inv_tsfm=None, device="cpu", vis=False, img_dir=""):

    num_batches = len(dataloader)

    SBP_test_loss = 0
    DBP_test_loss = 0

    MAE = 0
    mean = 0
    std = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            if tar_inv_tsfm:
                pred, y = tar_inv_tsfm(pred), tar_inv_tsfm(y)

            SBP_test_loss += loss_fn(pred[:,0], y[:,0]).item()
            DBP_test_loss += loss_fn(pred[:,1], y[:,1]).item()

            MAE += nn.L1Loss()(pred[:,0], y[:,0]).item()
            mean += np.mean((pred - y)[:,0].tolist())
            std += np.std((pred-y)[:,0].tolist())

    results = np.array([SBP_test_loss, DBP_test_loss, MAE, mean, std])

    print(f"Test Error:")
    print(f"SBP loss: {SBP_test_loss/num_batches:>5f}")
    print(f"DBP loss: {DBP_test_loss/num_batches:>5f}\n")

    if vis:
        utils.plot_test_distr(y[:,0].tolist(), pred[:,0].tolist(),
                              title="Predicted SBP vs True SBP", filename=img_dir+"predict_values")

    return results/num_batches

def train_full_test_once(train_dataloader, test_dataloader, model, loss_fn, optimizer=None, 
                      batch_size=128, epochs=200,
                      device="cpu", vis=False, img_dir=""):
    """
    Perform `epochs` loops of training and test the model once. Returns the final results of training 
    and testing.

    Args:
        training_dataloader:
            Training set dataloader.
        testing_data:
            Testing set dataloader.
        model:
            Machine learning model.
        loss_fn:
            Loss function of the model.
        optimizer:
            Optimizer of the model.
        batch_size (int):
            The size of samples fed into the network in each training iteration.
        epochs (int):
            The number of rounds of which the dataset is fed into the network.

    Returns:
        train_values (ndarray of shape (n_params, )):
            Training results.
        test_values (ndarray of shape (n_params, )):
            Testing results.
    """    


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_values = train_loop(train_dataloader, model, loss_fn, optimizer, device=device)

    test_values = test_loop(test_dataloader, model, loss_fn, device=device, vis=vis, img_dir=img_dir)

    return train_values, test_values


def train_test_scheme(training_data, testing_data, model, loss_fn, optimizer=None, 
                      batch_size=300, epochs=200, device="cpu"):
    """
    Perform `epochs` loops of training and testing. The training and testing results of each epoch 
    are stored in `train_history` and `test_history`.

    Args:
        training_data (PyTorch DataSet class):
            Training set.
        testing_data (PyTorch DataSet class):
            Testing set.
        model:
            Machine learning model.
        loss_fn:
            Loss function of the model.
        optimizer:
            Optimizer of the model.
        batch_size (int):
            The size of samples fed into the network in each training iteration.
        epochs (int):
            The number of rounds of which the dataset is fed into the network.

    Returns:
        train_history (ndarray of shape (epochs, n_params)):
            Training results.
        test_history (ndarray of shape (epochs, n_params)):
            Testing results.
    """    
    pin = False
    if device == "cuda":
        pin == True

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin)

    big_test_dataloader = DataLoader(testing_data, batch_size=len(testing_data), pin_memory=pin)

    train_history = np.zeros((epochs, 5))
    test_history = np.zeros((epochs, 5))

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_history[t,:] = train_loop(train_dataloader, model, loss_fn, optimizer, device=device)

        test_history[t,:] = test_loop(big_test_dataloader, model, loss_fn, device=device)

    for i in range(5):
        train_history[:,i] = utils.running_average(train_history[:,i])
        test_history[:,i] = utils.running_average(test_history[:,i])

    return train_history, test_history


def cross_validate_scheme(whole_dataset, model_class, loss_fn,
                          lr=0.01, batch_size=64, reg=0, epochs=10, k=5, 
                          device="cpu"):
    """
    Perform k-fold cross validation. The averaged training and validating results in each epoch 
    are stored in `train_results` and `valid_results`.

    Args:
        whole_dataset (PyTorch DataSet class):
            The entire dataset ready to split into training and testing sets.
        model_class:
            The class of the machine learning model.
        loss_fn:
            The loss function of the model.
        lr (float):
            Learning rate of the model.
        batch_size (int):
            The size of samples fed into the network in each training iteration.
        epochs (int):
            The number of rounds of which the dataset is fed into the network.

    Returns:
        train_results (ndarray of shape (epochs, n_params)):
            Training results.
        valid_results (ndarray of shape (epochs, n_params)):
            Validation results.
    """

    n = len(whole_dataset)
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    
    data, labels = whole_dataset[idx,:]

    train_results = np.zeros((epochs, 5))
    valid_results = np.zeros((epochs, 5))

    s_data = np.array_split(data, k, axis=0)
    s_labels = np.array_split(labels, k, axis=0)
    print(f"{k}-fold cross validation\n-------------------------------")
    for i in range(k):
        model = model_class().to(device)
        # Change the following line to use a different optimizer
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=reg)

        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=0)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=0)
        
        data_valid = s_data[i]
        labels_valid = s_labels[i]

        train_set = TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(labels_train).float())
        valid_set = TensorDataset(torch.from_numpy(data_valid).float(), torch.from_numpy(labels_valid).float())

        print(f"Fold {i+1}\n-------------------------------")
        train_values, valid_values = train_test_scheme(train_set, valid_set, model, loss_fn, 
                                                   optimizer=optimizer, 
                                                   batch_size=batch_size, 
                                                   epochs=epochs,
                                                   device=device)

        train_results += train_values
        valid_results += valid_values

    train_results, valid_results = train_results/k, valid_results/k

    return train_results, valid_results