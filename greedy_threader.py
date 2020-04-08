import torch
import torch.nn as nn
import data_loader
import network1 as network
import pandas as pd
import matplotlib.pyplot as plt
from vgg import vgg16, VGGNormLayer, perceptual_loss
from argparse import ArgumentParser

torch.backends.cudnn.benchmark = True
torch.autograd.set_grad_enabled(False)


def save_graph(path, val_losses, counter):
    '''
    Saves average perceptual loss given a certain number of patches
    '''
    plt.figure(figsize=(16, 8))
    plt.plot(counter, val_losses,
             color='blue')
    plt.scatter(counter, val_losses, color = 'blue')
    plt.suptitle('Network1 Perceptual Loss vs Number of 32x32 Patches')
    plt.legend(['Val Loss'], loc='upper right')
    plt.xlabel('Number of Patches')
    plt.ylabel('Perceptual loss')
    plt.savefig(path)


def patch_find_sing(net, patch_size, data, target, device, batch_size):
    '''
    With a greedy approach, determine where to place patches such that network can best reconstruct an image
    Return the centers of the patches and the associated loss
    '''
    points = [set() for _ in range(batch_size)]
    stored_points = [[] for _ in range(batch_size)]
    stored_losses = [[] for _ in range(batch_size)]
    x_crop = patch_size[0] // 2
    y_crop = patch_size[1] // 2
    # calculate 25 best locations for each image
    for _ in range(25):
        best_loss = [float('inf') for _ in range(batch_size)]
        best_tup = [(0, 0) for _ in range(batch_size)]
        for x in range(patch_size[0]//2, data.shape[2], patch_size[0]//2):
            for y in range(patch_size[1]//2, data.shape[3], patch_size[1]//2):
                old_mask = data[:, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop].clone()
                new_mask = torch.cat((target[:, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop],
                                      torch.ones((batch_size, 1) + patch_size).to(device)), dim=1)
                data[:, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = new_mask

                loss = net(data, target) #bx1
                for i in range(batch_size):
                    if (x, y) in points[i]:
                        continue
                    if loss[i] < best_loss[i]:
                        best_loss[i] = loss[i].item()
                        best_tup[i] = (x, y)
                data[:, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = old_mask
        for i in range(batch_size):
            x, y = best_tup[i]
            new_mask = torch.cat(
                (target[i:i + 1, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop],
                 torch.ones((1, 1) + patch_size).to(device)),
                dim=1)
            data[i:i + 1, :, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = new_mask
        for i in range(batch_size):
            stored_points[i].append(best_tup[i])
            stored_losses[i].append(best_loss[i])
            points[i].add(best_tup[i])

    return stored_points, stored_losses




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--gpus", type=int, nargs='+', default=list(range(torch.cuda.device_count())), help="gpus")

    parser.add_argument("--psize", type=int, default=32, help="patch size")

    parser.add_argument("--bvsize", type=int, default=32, help="batch size val")

    parser.add_argument("--dataset", type=str, default='celeba', help="name of dataset")

    parser.add_argument("--datapath", type=str, help="path to data set")

    parser.add_argument("--modelpath", type=str, help="path to pre-trained model weight")

    opt = parser.parse_args()

    device = torch.device("cuda:{}".format(opt.gpus[0]) if torch.cuda.is_available() else "cpu")
    device_ids, patch_size, batch_size = opt.gpus, (opt.psize, opt.psize), opt.bvsize

    # initialize trained network
    cnet = network.Net()
    cnet.load_state_dict(torch.load(opt.modelpath, map_location=device)['state_dict'])
    loss_net = network.FullModel(network.Net(), perceptual_loss, vgg16(), VGGNormLayer())
    if torch.cuda.device_count() > 1:
        loss_net = nn.DataParallel(loss_net, device_ids=device_ids)
    loss_net.to(device)


    # data loader
    if opt.dataset == 'celeba':
        loader = data_loader.get_celeba_loader
    elif opt.dataset == 'FFHQ':
        loader = data_loader.get_ffhq_loader
    else:
        loader = data_loader.get_stl_loader
    val_loader = loader(opt.datapath, patch_size, batch_size, "val")

    dic = {}
    results = []
    num_patches_loss = [0 for _ in range(25)]
    # this loop determines finds the optimal locations for patches for every image
    for ind, (data, target) in enumerate(val_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        stored_points, stored_losses = patch_find_sing(loss_net, patch_size, data, target, device, batch_size)
        for i in range(batch_size):
            results.append(stored_points[i])
            results.append(stored_losses[i])
            num_patches_loss = [i + j for i, j in zip(num_patches_loss, stored_losses[i])]
            dic[1+batch_size * ind + i] = (stored_points[i], stored_losses[i])
        print('Loss {}\n'.format(stored_losses[-1][-1]))
        torch.save(dic, "results/network1/patch_losses60.pickle")
        pd.DataFrame(results).to_csv('results/network1/network1_greedy60_losses.csv', index=False)
        save_graph("results/network1/network1_greedy60_graph.png", [i / (ind+1)/batch_size for i in num_patches_loss],
                   list(range(1, len(num_patches_loss) + 1)))

    # log results
    num_patches_loss = [i / len(val_loader.dataset) for i in num_patches_loss]
    with open("results/network1/network1_greedy60_log.txt", "w") as f:
        f.write("Network1 Greedy 60 epoch pretrained net for 32x32 patch.\n")
        for i, loss in enumerate(num_patches_loss):
            f.write("Num Patches: {}. Loss: {:4f}\n".format(i + 1, loss))
    torch.save(dic, "results/network1/img_patch_losses_map.pickle")
    pd.DataFrame(results).to_csv('results/network1/network1_greedy60_losses.csv', index=False)
    save_graph("results/network1/network1_greedy60_graph.png", num_patches_loss,
               list(range(1, len(num_patches_loss) + 1)))
