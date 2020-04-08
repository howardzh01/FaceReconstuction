import torch.nn as nn
from unet import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # start with 4 channels (3 for image and 1 for mask)
        self.inc = DoubleConv(4, 64, 4, padding=2, leaky_slope=0.2)
        self.down1 = DoubleConv(64, 128, 4, padding=2, stride=2, leaky_slope=0.2)
        self.down2 = DoubleConv(128, 256, 4, padding=2, stride=2, leaky_slope=0.2)
        self.down3 = DoubleConv(256, 256, 4, padding=2, stride=2, leaky_slope=0.2)
        self.down4 = DoubleConv(256, 256, 4, padding=2, stride=2, leaky_slope=0.2)
        self.down5 = DoubleConv(256, 256, 4, padding=2, leaky_slope=0.2)
        self.up0 = UpConv(512, 256, 4, padding=2, scale_factor=2)
        self.up1 = UpConv(512, 256, 4, padding=2, scale_factor=2)
        self.up2 = UpConv(512, 128, 4, padding=2, scale_factor=2)
        self.up3 = UpConv(256, 64, 4, padding=2, scale_factor=2)
        self.up4 = UpConv(128, 64, 4, padding=2, scale_factor=2)
        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class FullModel(nn.Module):
  def __init__(self, model, loss, loss_model, norm):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss
    self.loss_model = loss_model
    self.norm = norm

  def forward(self, input, target):
    output = self.model(input)
    loss = self.loss(output, target, self.loss_model, self.norm)
    return loss