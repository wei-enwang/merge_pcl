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
    def __init__(self, input_size, output_size, num_layers=3, dropout=0., device="cuda"):
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
        self.device = device
        self.output_size = output_size
    

    def forward_step(self, input_latent, hidden):
        """Helper function for forward below:
        Perform a single decoder step.

        Inputs:
        - `input_latent`: a 3d-tensor of shape (batch_size, 1, input_size)
            representing the padded embedded word vectors at this step in training
        - `hidden`: a 3d-tensor of shape (1, batch_size, output_size) representing
            the current hidden state.

        Returns:
        - `hidden`: a 3d-tensor of shape (1, batch_size, hidden_size)
            representing the current decoder hidden state.
        - `pre_output`: a 3d-tensor of shape (batch_size, 1, hidden_size)
            representing the total decoder output for one step
        """

        cur_output, hidden = self.rnn(input_latent, hidden)
        
        # cur_output, _ = torch.max(cur_output, dim=1, keepdim=True)

        return hidden, cur_output


    def forward(self, inputs, hidden=None, max_len=None):
        """
        Inputs:
        - `inputs`: a 3d-tensor of shape (batch_size, num_scenes, input_size)
            representing a batch of latencies of images.

        Returns:
        - `outputs`: a 3d-tensor of shape
            (batch_size, num_scenes, output_size).

        """
        batch_size = inputs.shape[0]
        # The maximum number of steps to unroll the RNN.
        if max_len is None:
            max_len = inputs.size(1)

        # Initialize decoder hidden state.
        if hidden is None:
            hidden = self.init_hidden(batch_size)


        outputs = []
        for i in range(max_len):
            hidden, output = self.forward_step(inputs[:,i:i+1,:], hidden)
            outputs.append(output[:,0,:])
        
        outputs = torch.stack(outputs, dim=1)

        return outputs

    
    def init_hidden(self, batch_size):
        """
        Input:
            - `batch_size`: a positive integer

        Returns:
            - `hidden`: a 2d-tensor of shape (batch_size, hidden_size) representing
                the initial hidden state of the RNN
        """
        # Use to initialize hidden state everytime before running a sentence.
        hidden = torch.zeros(batch_size, self.output_size).to(self.device)
        return hidden


class ShapeEncoder(nn.Module):
    """
    Wrapper model for the entire process from RGB-D images to shape latent.
    The last node of the output is confidence value
    """
    def __init__(self, latent_size=512, output_size=129, num_layers=3, dropout=0.):
        super().__init__()
        self.resnet = resnet18_4d(latent_size=latent_size)
        self.rnn = hidden_lstm(latent_size, output_size=output_size,
                               num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        hidden = self.resnet.forward(x)
        latents = self.rnn.forward(hidden)
        return latents
    
def loss_fn(z, gt_latent):
    conf = z[:,-1:]
    out_latent = z[:,:-1]
    
    assert z.shape[1] == gt_latent.shape[1]
    assert gt_latent.shape[0] == conf.shape[0]

    loss_1 = z[:,:,None]
    pass