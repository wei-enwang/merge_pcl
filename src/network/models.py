from typing import Type, Any, Callable, Union, List, Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torchvision import models

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        input_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(input_channels=3, **kwargs: Any) -> ResNet:
    """
    ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_channels, **kwargs)
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
        output = self.rnn(inputs)
        
        return output


class ShapeEncoder(nn.Module):
    """
    Wrapper model for the entire process from RGB-D images to shape latent.
    The last node of the output is confidence value
    """
    def __init__(self, latent_size=512, output_size=129, num_layers=1, num_obj=10, dropout=0.):
        super().__init__()
        self.num_obj = num_obj
        self.output_size = output_size
        self.cnn = resnet18(input_channels=4)
        self.rnn = hidden_lstm(latent_size, output_size=int(output_size*num_obj),
                               num_layers=num_layers, dropout=dropout)

    def forward_cnn(self, input):
        """
        Process the scenes sequentially with CNN.

        Inputss:
        - input (batch_size, seq_len, channels=4, image_height=224, image_width=224): Input images

        Returns:
        - x (batch_size, num_scenes, latent_size): The latents for each scene

        """
        seq_len = input.shape[1]
        enc_list = []
        for i in range(seq_len):
            enc = self.cnn(input[:,i,...])
            enc_list.append(enc)
        x = torch.stack(enc_list, dim=1)
        return x

    def forward(self, input) -> Tensor:
        """
        Inputs:
        - input (batch_size, num_scenes, channels=4, 
                 image_height=224, image_width=224): Input images

        Returns:
        - `outputs`: a 3d-tensor of shape
            (batch_size, num_scenes, num_obj, output_size).

        """
        b_size, num_scenes = input.shape[0], input.shape[1]
        assert input.shape[2] == 4
        assert input.shape[3] == 224
        assert input.shape[4] == 224
        hidden = self.forward_cnn(input)
        latents, _ = self.rnn(hidden)
        return latents.reshape((b_size, num_scenes, self.num_obj, self.output_size))


class Seq2SeqDecoder(nn.Module):
    """An RNN decoder without attention."""

    def __init__(self, input_size, hidden_size, num_layers=5, dropout=0.1):
        """
        Inputs:
            - `input_size`, `hidden_size`, and `dropout` the same as in Encoder.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = hidden_lstm(input_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward_step(self, prev_embed, hidden):
        """
        Perform a single decoder step.

        Inputs:
        - `prev_embed`: a 3d-tensor of shape (batch_size, 1, embed_size)
            representing the padded embedded word vectors at this step in training
        - `hidden`: a 3d-tensor of shape (1, batch_size, hidden_size) representing
            the current hidden state.

        Returns:
        - `hidden`: a 3d-tensor of shape (1, batch_size, hidden_size)
            representing the current decoder hidden state.
        - `pre_output`: a 3d-tensor of shape (batch_size, 1, hidden_size)
            representing the total decoder output for one step
        """
        # hint: you'll want to do more here than just run self.rnn (think about
        #       what you should do to the output of the self.rnn in order to
        #       compute the `pre_output`)

        ### Your code here!
        pre_output, hidden = self.rnn(prev_embed, hidden)
        
        pre_output, _ = torch.max(pre_output, dim=1, keepdim=True)

        return hidden, pre_output
        

    def forward(self, inputs, encoder_finals, hidden=None, max_len=None):
        """Unroll the decoder one step at a time.

        Inputs:
        - `inputs`: a 3d-tensor of shape (batch_size, max_seq_length, embed_size)
            representing a batch of padded embedded word vectors of target
            sentences (for teacher-forcing during training).
        - `encoder_finals`: a 3d-tensor of shape
            (num_enc_layers, batch_size, hidden_size) representing the final
            encoder hidden states used to initialize the initial decoder hidden
            states.
        - `hidden`: a 3d-tensor of shape (1, batch_size, hidden_size) representing
            the value to be used to initialize the initial decoder hidden states.
            If None, then use `encoder_finals`.
        - `max_len`: an int representing the maximum decoding length.

        Returns:
        - `hidden`: a 3d-tensor of shape
            (num_layers, batch_size, hidden_size) representing the final hidden
            state for each element in the batch.
        - `pre_output_vectors`: a 3d-tensor of shape
            (batch_size, max_len, hidden_size) representing the raw decoder
            outputs (before mapping to a `trg_vocab_size`-dim vector).
        """

        # The maximum number of steps to unroll the RNN.
        if max_len is None:
            max_len = inputs.size(1)

        # Initialize decoder hidden state.
        if hidden is None:
            hidden = self.init_hidden(encoder_finals)

        # hint: you'll want to keep track of the `pre_output` for each timestep,
        #       but you only need the final `hidden` state

        ### Your code here!
        pre_output_vectors = []
        for i in range(max_len):
            hidden, output = self.forward_step(inputs[:,i:i+1,:], hidden)
            pre_output_vectors.append(output[:,0,:])
        
        pre_output_vectors = torch.stack(pre_output_vectors, dim=1)

        return hidden, pre_output_vectors


    def init_hidden(self, encoder_finals):
        """Use encoder final hidden state to initialize decoder's first hidden
        state.

        Input: `encoder_finals` is same as in forward()

        Returns: 
            - `decoder_init_hiddens`: a 3d-tensor of shape 
                (num_layers, batch_size, hidden_size) representing the initial
                hidden state of the decoder for each element in the batch 
        """
        # hint: think about whether or not an activation function is needed here

        ### Your code here!
        decoder_init_hiddens=encoder_finals.repeat((3,1,1))
        
        return decoder_init_hiddens


class Seq2SeqShapeProcessor(nn.Module):
    """A standard Encoder-Decoder architecture without attention.
    """
    def __init__(self, latent_size=129, num_layers=5, dropout=0.1):
        """
        Inputs:
        - `encoder`: an `Encoder` object.
        - `decoder`: an `Decoder` object.
        """
        super().__init__()

        self.encoder = ShapeEncoder(output_size=latent_size, dropout=dropout)
        self.decoder = Seq2SeqDecoder(latent_size, latent_size, num_layers=num_layers, dropout=dropout)

    def forward(self, input_seq):
        """Take in and process masked source and target sequences.

        Inputs:
        `src_ids`: a 2d-tensor of shape (batch_size, max_seq_length) representing
            a batch of source sentences of word ids.
        `trg_ids`: a 2d-tensor of shape (batch_size, max_seq_length) representing
            a batch of target sentences of word ids.
        `src_lengths`: a 1d-tensor of shape (batch_size,) representing the
            sequence length of `src_ids`.

        Returns the decoder outputs, see the above cell.
        """
        x = self.encoder(input_seq)
        x = self.decoder(x)

        return x

    def encode(self, src_ids, src_lengths):
        return self.encoder(self.src_embed(src_ids), src_lengths)
        
    def decode(self, encoder_finals, trg_ids, decoder_hidden=None):
        return self.decoder(self.trg_embed(trg_ids), encoder_finals, decoder_hidden)
    

class crossMSEloss(nn.Module):
    """
    The loss function for this model.

    """
    def __init__(self, weight=None, size_average=True):
        super().__init__()
    

    def forward(self, inputs, targets):
        """
        Args:
        -inputs (batch_size, seq_len, num_preds, latent_size): The latents predicted by the LSTM network
        -targets (batch_size, seq_len, num_objects, latent_size): The ground truth latents

        Returns;
        - loss: A scalar that represents the average loss.
        """
        conf = inputs[...,-1]
        # confidence need to be positive
        conf = conf**2
        preds = inputs[...,:-1]

        # check latent size matches
        assert targets.shape[-1] == preds.shape[-1]
        assert preds.shape == targets.shape
        
        # dimension of the following term is (batch_size, seq_len, num_preds, num_objects)
        l2norm = torch.norm(preds[:,:,:,None,:] - targets[:,:,None,:,:], dim=-1)

        loss = torch.sum(conf*torch.min(l2norm, dim=-1)[0], dim=-1)+ \
               torch.sum(torch.min(1/conf[...,None]*l2norm, dim=2)[0], dim=-1)

        # The return value is the average loss over batch_size and seq_len
        return loss.mean()

