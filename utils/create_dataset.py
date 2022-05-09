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
    if name in _TORCH_BASIC_DS:
        ds_class = _TORCH_BASIC_DS[name]
        ds = ds_class(train=is_training, **torch_kwargs)

    elif filename != '':
        # custom dataset
        ds = TrainDataset(filename, root)

    return ds

