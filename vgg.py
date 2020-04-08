import torch
from collections import namedtuple
from torchvision import models as tv


class vgg16(torch.nn.Module):
    def __init__(self, n_return=5, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.n_return = n_return


    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        out = (h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out[0:self.n_return]


class VGGNormLayer(torch.nn.Module):
    def __init__(self):
        super(VGGNormLayer, self).__init__()
        self.register_buffer('mean', 
                             torch.Tensor([.485,.456,.406])
                             [None,:,None,None])
        self.register_buffer('std', 
                             torch.Tensor([.229,.224,.225])
                             [None,:,None,None])

    def forward(self, inp):
        return (inp - self.mean) / self.std



def perceptual_loss(x, y, net, norm=None):
    if norm is not None:
        x = norm(x)
        y = norm(y)

    f1 = net(x)
    f2 = net(y)
    n_layers = min(len(f1), len(f2))
    loss = []

    for l in range(n_layers):
        f1_l = f1[l]
        f2_l = f2[l]

        norm1 = torch.sqrt(torch.sum(f1_l * f1_l, dim=1, keepdim=True) + 1e-10)  # b x 1 x h x w
        norm2 = torch.sqrt(torch.sum(f2_l * f2_l, dim=1, keepdim=True) + 1e-10)

        f1_l_norm = torch.div(f1_l, norm1)  # b x c x h x w
        f2_l_norm = torch.div(f2_l, norm2)

        d = torch.mean(torch.sum((f1_l_norm - f2_l_norm)**2, dim=1), dim=(1, 2))  # bx1

        loss = d if l == 0 else loss + d
    return loss/n_layers
