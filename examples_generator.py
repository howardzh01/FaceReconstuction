import torch
import matplotlib.pyplot as plt
import data_loader
import numpy as np
import torchvision.transforms as T
to_img = T.ToPILImage()
import network1 as network
from vgg import vgg16, VGGNormLayer, perceptual_loss
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--psize", type=int, default=32, help="patch size")

parser.add_argument("--image_ids", type=int, nargs='+', help="image ids to display")

parser.add_argument("--dataset", type=str, default='celeba', help="name of data set")

parser.add_argument("--datapath", type=str, help="path to data set")

parser.add_argument("--modelpath", type=str, default='model_weights/celeba_network1_60eps_model.pth', help="path to pre-trained weights")

parser.add_argument("--dicpath", type=str, default='model_weights/celeba_patch_losses_map.pickle', help="path to patches dictionary")

opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
patch_size = (opt.psize, opt.psize)

vgg_net = vgg16().to(device)
vgg_norm = VGGNormLayer().to(device)
cn = network.Net().to(device)
cn.load_state_dict(torch.load(opt.modelpath, map_location=device)['state_dict'])
patched_dic = torch.load(opt.dicpath)
print(patched_dic.keys())
# data loader
if opt.dataset == 'celeba':
    loader = data_loader.CelebA
    factor = 28001
elif opt.dataset == 'FFHQ':
    loader = data_loader.FFHQ
    factor = 1
else:
    loader = data_loader.STL10
    factor = 1
img_loader = loader(opt.datapath, T.ToTensor(), patch_size, 'val', tuple(), None)


def modify_img(img, target, centers, num):
    for i in range(num):
        x, y = centers[i]
        x_crop, y_crop = patch_size
        img[:, 0:3, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = target[:, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop]
        img[:, 3:4, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = torch.ones((1,1,x_crop*2, y_crop*2)).to(device)
    return img


def load_img(img_id):
    '''
    assumes first image has id 0
    '''
    return img_loader[img_id-factor]


def im_choose_show(id_arr):
    '''
    Given list of image ids,
    '''
    fig = plt.figure(figsize=(20, 20))
    for i, img_id in enumerate(id_arr):
        inp, target = load_img(img_id)
        inp, target = inp.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
        centers = patched_dic[img_id][0]
        losses = patched_dic[img_id][1]
        for ind, j in enumerate([1, 2, 3, 5, 10, 20, 25]):
            img = modify_img(inp.clone(), target, centers, j)
            output = cn(img)
            plt.subplot(len(id_arr)*2, 8, 16 * i + ind + 1)
            if i == 0:
                plt.title("Patches = {}\nLoss: {:.3f}, {:.3f}".format(j, losses[j - 1],
                                                                      perceptual_loss(output, target, vgg_net,
                                                                                      vgg_norm).item()))
            else:
                plt.title("Loss: {:.3f}, {:.3f}".format(losses[j - 1],
                                                        perceptual_loss(output, target, vgg_net, vgg_norm).item()))
            output = output[0].data.cpu().numpy().clip(0, 1)
            plt.imshow(np.moveaxis(output, 0, -1), interpolation='none')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(len(id_arr)*2, 8, 16 * i + ind + 9)
            plt.imshow(to_img(img[0][0:4].cpu()), interpolation='none')
            plt.xticks([])
            plt.yticks([])
        plt.subplot(len(id_arr)*2, 8, 16 * i + 8)
        plt.imshow(to_img(target[0].cpu()), interpolation='none')
        plt.title("Target")
        plt.xticks([])
        plt.yticks([])
    fig.get_axes()[0].annotate("Title", (0.5, 0.95), xycoords='figure fraction', ha='center', fontsize=18)
    plt.show()
    # plt.savefig('examples.png', bbox_inches='tight')


if __name__ == '__main__':
    im_choose_show(opt.image_ids)
