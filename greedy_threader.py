import torch
import torch.nn as nn
import celeba_data_loader_multirand as celeba_data_loader
import network1 as network
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from vgg import vgg16, VGGNormLayer, perceptual_loss
torch.backends.cudnn.benchmark = True
torch.autograd.set_grad_enabled(False)


def save_graph(path, val_losses, counter):
    plt.figure(figsize=(16, 8))
    plt.plot(counter, val_losses,
             color='blue')
    plt.scatter(counter, val_losses, color = 'blue')
    plt.suptitle('CelebA Perceptual Loss vs Number of 32x32 Patches')
    plt.legend(['Val Loss'], loc='upper right')
    plt.xlabel('Number of Patches')
    plt.ylabel('Perceptual loss')
    plt.savefig(path)


def patch_find_sing(net, patch_size, data, target, device, batch_size):
    points = [set() for _ in range(batch_size)]
    stored_points = [[] for _ in range(batch_size)]
    stored_losses = [[] for _ in range(batch_size)]
    x_crop = patch_size[0] // 2
    y_crop = patch_size[1] // 2
    count = 0
    t1 = time()
    for _ in range(25):
        best_loss = [float('inf') for _ in range(batch_size)]
        best_tup = [(0, 0) for _ in range(batch_size)]
        for x in range(16, 256, 16):
            for y in range(16, 256, 16):
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
        count += 1
        if count % 5 == 0:
            print("Progress for batch is {}%. Minutes elapsed is {}\n".format(count*4, (time() - t1) / 60))

    return stored_points, stored_losses




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    t0 = time()
    batch_size = 100
    patch_size = (32, 32)
    cnet = network.Net()
    cnet.load_state_dict(torch.load('results/network1/network1_60eps_model.pth', map_location=device)['state_dict'])
    vgg_net = vgg16()
    vgg_norm = VGGNormLayer()
    loss_net = network.FullModel(cnet, perceptual_loss, vgg_net, vgg_norm, is_greedy=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", len(device_ids), "GPUs!")
        loss_net = nn.DataParallel(loss_net, device_ids=device_ids)
    loss_net.to(device)
    f = open("results/network1/network1_greedy60_log.txt", "w")
    f.write("Network1 Greedy 60 epoch pretrained net for 32x32 patch.\n")
    val_loader = celeba_data_loader.get_img_loader("/data/vision/billf/scratch/balakg/celeba_hq/images_256x256",
                                                   patch_size, batch_size, "val", shuffle = False)

    dic = {}
    results = []
    num_patches_loss = [0 for _ in range(25)]
    for ind, (data, target) in enumerate(val_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        stored_points, stored_losses = patch_find_sing(loss_net, patch_size, data, target, device, batch_size)
        for i in range(batch_size):
            results.append(stored_points[i])
            results.append(stored_losses[i])
            num_patches_loss = [i + j for i, j in zip(num_patches_loss, stored_losses[i])]
            dic[28001 + batch_size * ind + i] = (stored_points[i], stored_losses[i])
        print('\nProgress is {:.2f}%. Minutes elapsed is {}\n'.format(100*(ind+1)/len(val_loader), (time() - t0) / 60))
        print('Loss {}\n'.format(stored_losses[-1][-1]))
        if (ind +1) % 1 == 0:
            torch.save(dic, "results/network1/patch_losses60.pickle")
            pd.DataFrame(results).to_csv('results/network1/network1_greedy60_losses.csv', index=False)
            save_graph("results/network1/network1_greedy60_graph.png", [i / (ind+1)/batch_size for i in num_patches_loss],
                       list(range(1, len(num_patches_loss) + 1)))

    num_patches_loss = [i / len(val_loader.dataset) for i in num_patches_loss]
    for i, loss in enumerate(num_patches_loss):
        f.write("Num Patches: {}. Loss: {:4f}\n".format(i + 1, loss))
    torch.save(dic, "results/network1/img_patch_losses_map.pickle")
    pd.DataFrame(results).to_csv('results/network1/network1_greedy60_losses.csv', index=False)
    save_graph("results/network1/network1_greedy60_graph.png", num_patches_loss,
               list(range(1, len(num_patches_loss) + 1)))