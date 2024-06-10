"""
File name: unetSimple.py
Description: This script is used to train the UNET model on the golf dataset.

Authors: 

Roman Sabawoon Sekandari
Frederik Hoffmann Bertelsen
"""

import torch 
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.nn.functional as f


class DoubleConv(nn.Module):
    """
    DoubleConv module consists of two consecutive convolutional layers with batch normalization and ReLU activation.
    In each step of the U-Net model the DoubleConv module is used
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv module.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the DoubleConv module.
        """
        return self.conv(x)


class UNET(nn.Module):
    """
    UNET model for image segmentation.
    This models features and layers will be changed according to what kind of models we want to use.
    The models we use:


    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        features (list): List of number of features for each layer in the encoder and decoder. Main model is [64, 128, 256, 512].

    Attributes:
        downs: List of downsample layers in the encoder.
        ups: List of upsample layers in the decoder.
        pool: Max pooling layer.
        bottleneck (DoubleConv): Bottleneck layer.
        final_conv: Output convolutional layer.

    """

    def __init__(
            self, in_channels=3, out_channels=1, features=[32, 64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part of the UNET model
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder part of the UNET model
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck layer
        # -1 because it's the last element of the features list
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Output layer. Binary classification so out_channels = 1
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the UNET model.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # The skip connections are stored in a list
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()

