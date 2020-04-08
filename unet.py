import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    # conv -> batch norm -> leaky-relu
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, leaky_slope=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, scale_factor=2):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding=padding)  # normal Relu

    def forward(self, x1, x2):
        # x1 is prev image x2 is copy
        x1 = self.up(x1)

        x = torch.cat((x1, x2), dim=1)  # why dim=1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)