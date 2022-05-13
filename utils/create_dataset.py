import os
import glob
import torchvision

import torch.utils.data as data
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST


_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    fashion_mnist=FashionMNIST
)


def create_dataset(
        name,
        root,
        filename=None,
        is_training=True,
        download=False,
        **kwargs
):

    name = name.lower()
    torch_kwargs = dict(root=root, download=download, **kwargs)
    print(name)
    if name in _TORCH_BASIC_DS:
        ds_class = _TORCH_BASIC_DS[name]
        ds = ds_class(train=is_training, **torch_kwargs)

    elif filename != '':
        # custom dataset
        ds = CustomDataset(root, **kwargs)

    return ds


def read_file(filename):
    image_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip().split(' ')
            name = content[0]
            image_list.append(name)
        return image_list


class CustomDataset(data.Dataset):
    def __init__(self, image_dir, transform):
        self.fnames = glob.glob(os.path.join(image_dir, '*'))
        self.transform = transform
        self.len = len(self.fnames)

    def __getitem__(self, idx):
        i = idx % self.len
        img = torchvision.io.read_image(self.fnames[i])
        return self.transform(img)

    def __len__(self):
        return self.len


