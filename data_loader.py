import torch
from torch.utils import data
from torchvision import transforms as T
import os
from PIL import Image
import random
import glob


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, patch_size, mode, points, num_rand):
        self.image_dir = image_dir
        self.transform = transform
        self.patch_size = patch_size
        self.mode = mode  # train, val or test
        self.preprocess()
        self.points = points
        self.num_rand = num_rand

    def preprocess(self):
        """Preprocess attribute file. Store filenames and labels."""
        lines = [_ for _ in sorted(os.listdir(self.image_dir)) if _[-4:] == ".jpg"]

        n_total = len(lines)
        n_val = 1000
        n_test = 1000
        n_train = n_total - n_val - n_test

        if self.mode == "train":
            self.dataset = lines[0:n_train]
        elif self.mode == "val":
            self.dataset = lines[n_train:n_train + n_val]
        else:
            self.dataset = lines[n_train + n_val:]

    def __getitem__(self, index):
        filename = self.dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        out = self.transform(image)
        if self.num_rand:
            mask = generate_rand_mask(out, self.patch_size, random.randrange(1, self.num_rand+1))
        else:
            mask = generate_mask(out, self.patch_size, self.points)
        inp = torch.cat((out*mask, mask))
        return inp, out

    def __len__(self):
        return len(self.dataset)


class STL10(data.Dataset):

    def __init__(self, image_dir, transform, patch_size, mode, points, num_rand):
        self.image_dir = image_dir
        self.transform = transform
        self.patch_size = patch_size
        self.mode = mode  # train, val or test
        self.preprocess()
        self.points = points
        self.num_rand = num_rand

    def preprocess(self):
        self.files = glob.glob(os.path.join(self.image_dir, self.mode + '/*.jpg'))

    def __getitem__(self, index):
        out = self.transform(Image.open(self.files[index]))
        if self.num_rand:
            mask = generate_rand_mask(out, self.patch_size, random.randrange(1, self.num_rand + 1))
        else:
            mask = generate_mask(out, self.patch_size, self.points)
        inp = torch.cat((out * mask, mask))
        return inp, out

    def __len__(self):
        return len(self.files)


class FFHQ(data.Dataset):
    def __init__(self, data_path, transform, patch_size, mode, points, num_rand):
        self.data_path = data_path
        self.transform = transform
        self.patch_size = patch_size
        self.mode = mode  # train, val or test
        self.preprocess()
        self.points = points
        self.num_rand = num_rand

    def preprocess(self):
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.data_path):
            for file in f:
                if '.png' in file:
                    files.append(os.path.join(r, file))
        files.sort()
        if self.mode == "train":
            files = files[0:60000]
        elif self.mode == "val":
            files = files[60000:65000]
        else:
            files = files[65000:]
        self.files = files

    def __getitem__(self, index):
        im = Image.open(self.files[index])
        out = self.transform(im)
        if self.num_rand:
            mask = generate_rand_mask(out, self.patch_size, random.randrange(1, self.num_rand+1))
        else:
            mask = generate_mask(out, self.patch_size, self.points)
        inp = torch.cat((out * mask, mask))
        return inp, out

    def __len__(self):
        return len(self.files)


def generate_rand_mask(out, patch_size, num_patches):
    shape = out.shape
    mask = torch.zeros(1, shape[1], shape[2])
    x_crop = patch_size[0] // 2
    y_crop = patch_size[1] // 2
    for _ in range(num_patches):
        x = random.randrange(1 + x_crop, shape[1] - x_crop)
        y = random.randrange(1 + y_crop, shape[2] - y_crop)
        mask[:, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = torch.ones(patch_size)
    return mask


def generate_mask(out, patch_size, points):
    shape = out.shape
    mask = torch.zeros(1, shape[1], shape[2])
    x_crop = patch_size[0] // 2
    y_crop = patch_size[1] // 2
    for x, y in points:
        mask[:, x - x_crop:x + x_crop, y - y_crop:y + y_crop] = torch.ones(patch_size)
    return mask


def get_celeba_loader(image_dir, image_size, batch_size, mode, points=tuple(), num_rand=None):
    dataset = CelebA(image_dir, T.ToTensor(), image_size, mode, points, num_rand)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == "train"), pin_memory=True)


def get_stl_loader(image_dir, image_size, batch_size, mode, points=tuple(), num_rand=None):
    dataset = STL10(image_dir, T.ToTensor(), image_size, mode, points, num_rand)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == "train"), pin_memory=True)


def get_ffhq_loader(data_path, patch_size, batch_size, mode, points=tuple(), num_rand=None):
    transform = []
    transform.append(T.CenterCrop((896, 640)))
    transform.append(T.Resize((224, 160)))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)
    dataset = FFHQ(data_path, transform, patch_size, mode, points, num_rand)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == "train"), pin_memory=True)



