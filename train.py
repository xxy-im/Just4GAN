import yaml
import argparse

from utils import create_dataset, create_model

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
    generator, discriminator = create_model(config.model)
    G = generator(config.in_feat, config.img_shape, init_weights=False)
    D = discriminator(config.img_shape, init_weights=False)
    G.to(device)
    D.to(device)

    optim_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))

    loss_fn = torch.nn.BCELoss()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
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
    for i, (images, _) in enumerate(pbar):
        images = images.to(device)
        bs = images.shape[0]
        z = torch.randn((bs, 128), device=device)
        gz = G(z)
        real = torch.ones((bs, 1), device=device)
        fake = torch.zeros((bs, 1), device=device)

        r_loss = loss_fn(D(images), real)  # 识别真实图片的loss
        f_loss = loss_fn(D(gz), fake)  # 识别假图片的loss
        D_loss = (r_loss + f_loss) / 2  # 取平均

        D.zero_grad()
        D_loss.backward()
        optim_D.step()

        # 训练k次D后训练G
        if i % k == 0:
            G_loss = loss_fn(D(G(z)), real)
            G.zero_grad()
            G_loss.backward()
            optim_G.step()

        pbar.set_postfix(
            {'D Loss': '{:.5f}'.format(round(D_loss.item(), 4)), 'G Loss': '{:.5f}'.format(round(G_loss.item(), 4))})


if __name__ == '__main__':
    main()
