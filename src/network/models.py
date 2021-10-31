import numpy as np
import torch
from torch import nn
from torchvision import models

def resnet18_4d(model = models.resnet18(pretrained=True), new_in_channels=4, latent_size=512):

    layer = model.conv1
            
    # Creating new Conv2d layer
    new_layer = nn.Conv2d(in_channels=new_in_channels, 
                    out_channels=layer.out_channels, 
                    kernel_size=layer.kernel_size, 
                    stride=layer.stride, 
                    padding=layer.padding,
                    bias=layer.bias)

    copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights

    # Copying the weights from the old to the new layer
    new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

    #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
    for i in range(new_in_channels - layer.in_channels):
        channel = layer.in_channels + i
        new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
    new_layer.weight = nn.Parameter(new_layer.weight)

    model.conv1 = new_layer

    # modify output layer
    model.fc = nn.Linear(512, latent_size)

    return model


class hidden_lstm(nn.Module):
    def __init__(self, input_size, output_size, num_layers=3, dropout=0.):
        """
        Inputs: 
        - `input_size`: an int representing the RNN input size.
        - `output_size`: an int representing the RNN hidden size.
        - `dropout`: a float representing the dropout rate during training. Note
            that for 1-layer RNN this has no effect since dropout only applies to
            outputs of intermediate layers.
        """
        super(hidden_lstm, self).__init__()
        self.rnn = nn.LSTM(input_size, output_size, num_layers=num_layers, batch_first=True,
                        dropout=dropout, bidirectional=False)
        self.num_layers = num_layers
        self.output_size = output_size


    def forward(self, inputs):
        """
        Inputs:
        - `inputs`: a 3d-tensor of shape (batch_size, num_scenes, input_size)
            representing a batch of latencies of images.

        Returns:
        - `outputs`: a 3d-tensor of shape
            (batch_size, num_scenes, output_size(default=10*129)).

        """
        batch_size = inputs.shape[0]

        seq_len = inputs.size(1)

        output = self.rnn(inputs)
        
        return output.reshape((batch_size, seq_len, 10, 129))


class ShapeEncoder(nn.Module):
    """
    Wrapper model for the entire process from RGB-D images to shape latent.
    The last node of the output is confidence value
    """
    def __init__(self, latent_size=128, output_size=1290, num_layers=3, dropout=0.):
        super().__init__()
        self.resnet = resnet18_4d(latent_size=latent_size)
        self.rnn = hidden_lstm(latent_size, output_size=output_size,
                               num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        hidden = self.resnet.forward(x)
        latents = self.rnn.forward(hidden)
        return latents
    

class crossMSEloss(nn.Module):
    """
    The loss function for this model.

    """
    def __init__(self, weight=None, size_average=True):
        super().__init__()
    

    def forward(self, inputs, targets):
        """
        inputs (batch_size, seq_len, num_preds, latent_size): The latents predicted by the LSTM network
        targets (batch_size, seq_len, num_objects, latent_size): The ground truth latents
        """
        conf = inputs[...,-1]
        preds = inputs[...,:-1]

        # check latent size matches
        assert targets.shape[-1] == preds.shape[-1]
        assert inputs.shape == targets.shape
        
        # dimension of the following term is (batch_size, seq_len, num_preds, num_objects)
        l2norm = torch.norm(preds[:,:,:,None,:] - targets[:,:,None,:,:], dim=-1)

        loss = torch.sum(conf*torch.min(l2norm, dim=-1)[0], dim=-1)+ \
               torch.sum(torch.min(1/conf[...,None]*l2norm, dim=2)[0], dim=-1)

        return loss

