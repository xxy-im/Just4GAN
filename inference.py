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

parser.add_argument("--device", type=str, help="your cuda device", default='cuda:0')

parser.add_argument('--ckpt', type=str, metavar='checkpoint', help='path to checkpoint')

parser.add_argument('--in-feat', type=int, default=128, metavar='N',
                    help='The length of random noise (default: 128)')

parser.add_argument('--n-output', type=int, default=1000, metavar='N',
                    help='The nums of output images (default: 1000)')


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
    z_sample = torch.randn((config.n_output, config.in_feat), device=device)
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(config.out_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # TODO: 加点输出日志


if __name__ == '__main__':
    main()
