import os
import yaml
import argparse

from utils import  create_model

import torch
import torchvision

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

    device = torch.device(config.device)

    generator, _ = create_model(config.model)
    G = generator(config.in_feat, config.img_shape, False)
    G.load_state_dict(torch.load(config.inference_ckpt))
    G.to(device)
    G.eval()

    # 默认输出1000张照片到out_dir/result.jpg
    n_output = 1000
    z_sample = torch.randn((n_output, config.in_feat), device=device)
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(config.out_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # TODO: 加点输出日志


if __name__ == '__main__':
    main()
