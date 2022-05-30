import yaml
import argparse
import random

import models
from utils import create_dataset

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from tqdm import tqdm

# reference: https://github.com/rwightman/pytorch-image-models/blob/master/train.py
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser.add_argument("--epochs", type=int, help="training epochs", default=100)
parser.add_argument("--batch_size", type=int, help="data batch size", default=128)
parser.add_argument("--device", type=str, help="your cuda device", default='cuda:0')

# dataset config
parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: cifar10 if empty)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for CIFAR100, CIFAR10, MNIST, FashionMNIST.')

# Model parameters
parser.add_argument('--model', default='gan', type=str, metavar='MODEL', help='default: "vanilla-gan"')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--in-feat', type=int, default=128, metavar='N',
                    help='The length of random noise (default: 128)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.05)')


def _parse_args():
    args_config, _ = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


def main():
    config = _parse_args()
    start_epoch = 0
    n_epochs = config.epochs
    weights_folder = config.ckpt_dir

    device = torch.device(config.device)
    generator, discriminator = models.create_model(config.model)
    G = generator(config.in_feat, config.img_shape, init_weights=False)
    D = discriminator(config.img_shape, init_weights=False)
    G.to(device)
    D.to(device)

    optim_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optim_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    loss_fn = torch.nn.BCELoss()

    transform = [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    if config.model == 'dcgan':
        transform.insert(0, transforms.Resize((64, 64)))

    transform = transforms.Compose(transform)
    train_data = create_dataset(config.dataset, config.data_dir, download=config.dataset_download, transform=transform)
    train_loader = data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    if config.resume:
        pass
        # TODO: load_dict
        # start_epoch = load里的

    for epoch in range(start_epoch, n_epochs):
        train_one_epoch(config, epoch, G, D, optim_G, optim_D, train_loader, loss_fn, device)
        torch.save(G.state_dict(), f'{weights_folder}/G-checkpoint-{str(epoch).zfill(3)}epoch.pth')
        torch.save(D.state_dict(), f'{weights_folder}/D-checkpoint-{str(epoch).zfill(3)}epoch.pth')
        # TODO: save里保存epoch信息
        # TODO: 增加tensorboard 或 wandb 配置


def train_one_epoch(config, epoch, G, D, optim_G, optim_D, train_loader, loss_fn, device):
    k = config.kstep

    G.train()
    D.train()

    pbar = tqdm(train_loader)
    pbar.set_description(desc=f'Epoch {epoch + 1}/{config.epochs}')

    # cifar-10是带标注的，所有加了个 _ 丢弃标注
    for i, (images, _) in enumerate(pbar):
        r_images = images.to(device)
        bs = r_images.shape[0]
        z = torch.randn((bs, config.in_feat), device=device)
        f_images = G(z)
        real = torch.ones((bs, 1), device=device)
        fake = torch.zeros((bs, 1), device=device)

        r_loss = loss_fn(D(r_images.detach()), real)  # 识别真实图片的loss
        f_loss = loss_fn(D(f_images.detach()), fake)  # 识别假图片的loss
        D_loss = (r_loss + f_loss) / 2  # 取平均

        D.zero_grad()
        D_loss.backward()
        optim_D.step()

        # 训练k次D后训练G
        if i % k == 0:
            G_loss = loss_fn(D(f_images), real)
            G.zero_grad()
            G_loss.backward()
            optim_G.step()

        pbar.set_postfix(
            {'D Loss': '{:.5f}'.format(round(D_loss.item(), 4)), 'G Loss': '{:.5f}'.format(round(G_loss.item(), 4))})


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    same_seeds(42)      # 宇宙的答案 42
    main()
