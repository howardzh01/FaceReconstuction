import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data_loader
import network1 as network
from vgg import vgg16, VGGNormLayer, perceptual_loss
from argparse import ArgumentParser

torch.backends.cudnn.benchmark = True
log_interval = 200


def train(net, optimizer, train_loader, epoch, device):
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = torch.mean(net(data, target))
        loss.backward()  # computes gradients
        optimizer.step()  # updates parameters
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print('\nTrain set: Avg. loss: {:.4f}\n'.format(train_loss))
    return train_loss


def val(net, val_loader, device):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            val_loss += torch.mean(net(data, target)).item()
    val_loss /= len(val_loader)
    print('\nVal set: Avg. loss: {:.4f}\n'.format(val_loss))
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
    parser = ArgumentParser()

    parser.add_argument("--gpus", type=int, nargs='+', default=list(range(torch.cuda.device_count())), help="gpus")

    parser.add_argument("--eps", type=int, default=20, help="number of epochs")

    parser.add_argument("--psize", type=int, default=32, help="patch size")

    parser.add_argument("--btsize", type=int, default=32, help="batch size train")

    parser.add_argument("--bvsize", type=int, default=32, help="batch size val")

    parser.add_argument("--dataset", type=str, default='celeba', help="name of dataset")

    parser.add_argument("--datapath", type=str, help="path to data set")

    opt = parser.parse_args()

    device = torch.device("cuda:{}".format(opt.gpus[0]) if torch.cuda.is_available() else "cpu")
    device_ids, n_epochs, patch_size = opt.gpus, opt.eps, (opt.psize, opt.psize),
    batch_size_train, batch_size_test = opt.btsize, opt.bvsize

    # data loader
    if opt.dataset == 'celeba':
        loader = data_loader.get_celeba_loader
    elif opt.dataset == 'FFHQ':
        loader = data_loader.get_ffhq_loader
    else:
        loader = data_loader.get_stl_loader
    train_loader = loader(opt.datapath, patch_size, batch_size_train, "train", num_rand=25)
    val_loader = loader(opt.datapath, patch_size, batch_size_test, "val", num_rand=25)

    # initialize network and optimizer
    net = network.FullModel(network.Net(), perceptual_loss, vgg16(), VGGNormLayer())
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # train/validation loop + log results
    best_loss = float('inf')
    train_losses, val_losses, counter = [], [], []
    val_loss = val(net, val_loader, device)
    with open("results/network1/network1_log.txt", "w") as f:
        f.write("network1 perceptual loss \n")
        f.write("train on 25 random 12x12 patches\n")
        f.write(f"{val_loss}")
    for epoch in range(1, n_epochs + 1):
        train_loss = train(net, optimizer, train_loader, epoch, device)
        train_losses.append(train_loss)
        val_loss = val(net, val_loader, device)
        val_losses.append(val_loss)
        counter.append(epoch)
        with open("results/network1/network1_log.txt", "a+") as f:
            f.write('\nTrain Epoch: {} \nTrain set: Avg. loss: {:.4f}'.format(epoch, train_loss))
            f.write('\nVal set: Avg. loss: {:.4f}\n'.format(val_loss))
            if epoch == n_epochs:
                f.write('\nEpoch Counter, Train Loss, Val Loss\n{}\n{}\n{}\n'.format(counter, train_losses, val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'epoch': epoch, 'state_dict': net.module.state_dict(), 'perceptual_loss': best_loss},
                       'results/network1/network1_model.pth')
        save_graph("results/network1/network1_train_loss_graph.png", train_losses, val_losses, counter)
