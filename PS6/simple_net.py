import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """Simple Network with atleast 2 conv2d layers and two linear layers."""

    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Hints:
        1. Refer to https://pytorch.org/docs/stable/nn.html for layers
        2. Remember to use non-linearities in your network. Network without
        non-linearities is not deep.
        3. You will get 4D tensor for an image input from self.conv_layers. You need 
        to process it and make it a compatible tensor input for self.fc_layers.
        """
        super().__init__()

        self.conv_layers = nn.Sequential()  # conv2d and supporting layers here
        self.fc_layers = nn.Sequential()  # linear and supporting layers here
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError("SimpleNet function not implemented")

        conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        bn1 = nn.BatchNorm2d(num_features=10)
        act1 = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3)

        conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        bn2 = nn.BatchNorm2d(num_features=20)
        act2 = nn.ReLU()
        pool2 = nn.MaxPool2d(kernel_size=3)

        fc3 = nn.Linear(in_features=720, out_features=100)
        act3 = nn.ReLU()
        fc4 = nn.Linear(in_features=100, out_features=15)

        self.conv_layers = nn.Sequential(conv1, bn1, act1, pool1, conv2, bn2, act2, pool2)
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

        # raise NotImplementedError("forward function of SimpleNet not implemented")

        conv_features = self.conv_layers(x)
        flattened_conv_features = conv_features.view(-1, 720)
        model_output = self.fc_layers(flattened_conv_features)

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
