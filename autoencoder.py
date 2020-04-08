import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 4, padding = 1, stride = 2)
        self.conv2 = nn.Conv2d(64, 128, 4, padding = 1, stride = 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding= 1, stride=2)
        self.conv4 = nn.Conv2d(256, 256, 3, padding = 1, stride=2)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.deconv1 = nn.Conv2d(256, 256, 3, padding = 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256, 128, 3, padding = 1)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.deconv4 = nn.Conv2d(64, 3, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.bn4(x)
        x = F.leaky_relu(self.deconv1(self.up1(x)), 0.2)
        x = self.bn5(x)
        x = F.leaky_relu(self.deconv2(self.up2(x)), 0.2)
        x = self.bn6(x)
        x = F.leaky_relu(self.deconv3(self.up3(x)), 0.2)
        x = self.bn7(x)
        x = self.deconv4(self.up4(x))
        return x

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