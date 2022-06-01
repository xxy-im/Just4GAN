import random
import config

from datasets import AnimeDataset
from pix2pix import UNetGenerator, PatchDiscriminator

import numpy as np
import torch
from torch.utils import data

import torchvision
from torchvision import transforms

from tqdm import tqdm


def main():
    start_epoch = 0
    n_epochs = config.epochs
    weights_folder = config.ckpt_dir

    G, D = UNetGenerator(init_weights=True), PatchDiscriminator(init_weights=True)
    G.to(config.device)
    D.to(config.device)

    optim_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))

    bce_loss = torch.nn.BCEWithLogitsLoss()
    l1_loss = torch.nn.L1Loss()

    train_transform = [
        transforms.Resize((256, 256)),
        transforms.ToPILImage(),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    train_transform = transforms.Compose(train_transform)

    train_data = AnimeDataset(config.train_dir, train_transform)
    train_loader = data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_data = AnimeDataset(config.val_dir, train_transform)
    val_loader = data.DataLoader(val_data, batch_size=5)

    for epoch in range(start_epoch, n_epochs):
        train_one_epoch(epoch, G, D, optim_G, optim_D, train_loader, bce_loss, l1_loss)
        torch.save(G.state_dict(), f'{weights_folder}/G-checkpoint-{str(epoch).zfill(3)}epoch.pth')
        torch.save(D.state_dict(), f'{weights_folder}/D-checkpoint-{str(epoch).zfill(3)}epoch.pth')
        test(G, val_loader, epoch)
        # TODO: save里保存epoch信息
        # TODO: 增加tensorboard 或 wandb 配置


def train_one_epoch(epoch, G, D, optim_G, optim_D, train_loader, gan_loss, dist_loss):
    pbar = tqdm(train_loader)
    pbar.set_description(desc=f'Epoch {epoch + 1}/{config.epochs}')

    for i, (x, y) in enumerate(pbar):
        x, y = x.to(config.device), y.to(config.device)

        y_fake = G(x)   # 生成器生成假图
        d_real = D(x, y)    # 判别器对真图进行判别
        d_fake = D(x, y_fake.detach())   # 判别器对假图进行判别

        real = torch.ones_like(d_real)    # 真图全1
        fake = torch.zeros_like(d_real)   # 假图全0

        r_loss = gan_loss(d_real, real)
        f_loss = gan_loss(d_fake, fake)
        D_loss = (r_loss + f_loss) / 2

        # 训练判别器，不需要l1_loss
        D.zero_grad()
        D_loss.backward()
        optim_D.step()

        # 训练生成器
        d_fake = D(x, y_fake)
        G_gan_loss = gan_loss(d_fake, fake)
        l1_loss = dist_loss(y_fake, y)
        G_loss = G_gan_loss + config.l1_lambda * l1_loss
        G.zero_grad()
        G_loss.backward()
        optim_G.step()

        pbar.set_postfix(
            {'D Loss': '{:.5f}'.format(round(D_loss.item(), 4)), 'G Loss': '{:.5f}'.format(round(G_loss.item(), 4))})


def test(gen, val_loader, epoch):
    x, y = next(iter(val_loader))
    x, y = x.to(config.device), y.to(config.device)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5    # 消除正则
        x = x * 0.5 + 0.5
        output = torch.cat((x, y_fake), 3)
        torchvision.utils.save_image(output, config.output_dir + f"/output_{epoch}.png")

    gen.train()


def random_seeds(seed):
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
    random_seeds(42)      # 宇宙的答案 42
    main()
