import os
import argparse

import config
from pix2pix import UNetGenerator

import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="", help="the image you wanna input")
opt = parser.parse_args()


# 推理单张图片
def main():
    generator = UNetGenerator()
    generator.load_state_dict(torch.load(config.inference_ckpt))
    generator.to(config.device)
    generator.eval()

    x = torchvision.io.read_image(opt.image)

    with torch.no_grad():
        pred = generator(x)
        # pred = pred * 0.5 + 0.5  # 消除正则
        # x = x * 0.5 + 0.5
        output = torch.cat((x, pred), 3)
        torchvision.utils.save_image(output, config.output_dir + f"/output.png")

    # TODO: 加点输出日志


if __name__ == '__main__':
    main()
