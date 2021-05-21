import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError("SimpleNetDropout function not implemented")

        conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        bn1 = nn.BatchNorm2d(num_features=10)
        act1 = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3)

        drop1 = nn.Dropout(p=0.1)

        conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        bn2 = nn.BatchNorm2d(num_features=20)
        act2 = nn.ReLU()
        pool2 = nn.MaxPool2d(kernel_size=3)

        drop2 = nn.Dropout(p=0.1)

        fc3 = nn.Linear(in_features=720, out_features=100)
        act3 = nn.ReLU()
        fc4 = nn.Linear(in_features=100, out_features=15)

        self.conv_layers = nn.Sequential(conv1, bn1, act1, pool1, drop1, 
                                        conv2, bn2, act2, pool2, drop2)
        self.fc_layers = nn.Sequential(fc3, act3, fc4)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError("forward function of SimpleNetDropout not implemented")

        conv_features = self.conv_layers(x)
        flattened_conv_features = conv_features.view(-1, 720)
        model_output = self.fc_layers(flattened_conv_features)

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
