import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import celeba_data_loader_multirand as celeba_data_loader
import network1 as network
from time import time
from vgg import vgg16, VGGNormLayer, perceptual_loss
torch.backends.cudnn.benchmark = True


log_interval = 200
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device_ids = [2, 3, 4, 5, 6]
vgg_net = nn.DataParallel(vgg16().to(device), device_ids=device_ids)
vgg_norm = VGGNormLayer().to(device)

def train(net, optimizer, train_loader, epoch, device, f=None):
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = perceptual_loss(output, target, vgg_net, vgg_norm)
        loss.backward() #computes gradients
        optimizer.step() #updates parameters
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item()
    train_loss /= len(train_loader)
    if f is not None:
        f.write('\nTrain Epoch: {} \nTrain set: Avg. loss: {:.4f}'.format(epoch, train_loss))
        print('\nTrain set: Avg. loss: {:.4f}\n'.format(train_loss))
    return train_loss


def val(net, val_loader, device, f = None, num_batches=None):
    net.eval()
    val_loss = 0
    count = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = net(data)
            val_loss += perceptual_loss(output, target, vgg_net, vgg_norm)
            count += 1
            if num_batches is not None and num_batches == count:
                break
    val_loss /= count
    if f is not None:
        f.write('\nVal set: Avg. loss: {:.4f}\n'.format(
            val_loss))
    return val_loss

def save_graph(path, train_losses, val_losses, counter):
    plt.figure(figsize=(16, 8))
    plt.plot(counter, train_losses,
             color='blue')
    plt.scatter(counter, train_losses,
                color='blue')
    plt.scatter(counter, val_losses, color='red')
    plt.suptitle("Network1 Train/Val Loss for 32x32 patch")
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('number of epochs')
    plt.ylabel('Perceptual loss')
    plt.savefig(path)

if __name__ == '__main__':
    best_loss = float('inf')
    batch_size_train = 32
    batch_size_test = 32
    learning_rate = 1e-4
    n_epochs = 90
    t0 = time()
    patch_size = (32, 32)
    train_loader = celeba_data_loader.get_img_loader("/data/vision/billf/scratch/balakg/celeba_hq/images_256x256",  patch_size, batch_size_train, "train", num_rand = 25)
    val_loader = celeba_data_loader.get_img_loader("/data/vision/billf/scratch/balakg/celeba_hq/images_256x256",  patch_size, batch_size_test, "val", num_rand = 25)
    train_losses = []
    val_losses = []
    counter = []
    print(len(val_loader.dataset))  # 1000
    print(len(train_loader.dataset))  # 28000
    t1 = time()
    net = network.Net()
    if torch.cuda.device_count() > 1:
        print("Let's use", len(device_ids), "GPUs!")
        net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    f = open("results/network1/network1_log.txt", "w")
    f.write("network1 perceptual loss \n")
    f.write("train on 100 random 32x32 patches\n")
    val(net, val_loader, device, f)
    print("val time is {} minutes".format((time() - t1)/60))
    for epoch in range(1, n_epochs + 1):
        train_loss = train(net, optimizer, train_loader, epoch, device, f)
        train_losses.append(train_loss)
        loss = val(net, val_loader, device, f)
        val_losses.append(loss)
        counter.append(epoch)
        print('Minutes elapsed is {}'.format((time() - t1) / 60))
        print('\nVal set: Avg. loss: {:.4f}\n'.format(
            loss))
        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch, 'state_dict': net.module.state_dict(), 'perceptual_loss': best_loss}, 'results/network1/network1_model.pth')
    save_graph("results/network1/network1_train_loss_graph.png", train_losses, val_losses, counter)
    f.close()

