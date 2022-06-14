import random
import config
import itertools

from datasets import Paint2PhotoDataset
from cyclegan import ResidualGenerator, PatchDiscriminator

import numpy as np
import torch
from torch.utils import data

import torchvision
from torchvision import transforms

from tqdm import tqdm


def main():
    start_epoch = 0
    n_epochs = config.n_epochs
    weights_folder = config.ckpt_dir

    G_AB = ResidualGenerator(res_nums=config.res_nums)
    G_BA = ResidualGenerator(res_nums=config.res_nums)
    D_A = PatchDiscriminator()
    D_B = PatchDiscriminator()

    G_AB.to(config.device)
    G_BA.to(config.device)
    D_A.to(config.device)
    D_B.to(config.device)

    optim_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                               lr=config.lr, betas=(0.5, 0.999))
    optim_DA = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optim_DB = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(0.5, 0.999))

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - config.n_epochs) / float(config.decay_epoch + 1)
        return lr_l

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=lambda_rule)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optim_DA, lr_lambda=lambda_rule)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optim_DB, lr_lambda=lambda_rule)

    gan_loss = torch.nn.MSELoss()
    cyc_loss = torch.nn.L1Loss()
    l_identity = torch.nn.L1Loss()

    train_transform = [
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    train_transform = transforms.Compose(train_transform)

    train_data = Paint2PhotoDataset(config.data_dir, train_transform)
    train_loader = data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                   num_workers=config.num_workers)

    val_data = Paint2PhotoDataset(config.data_dir, train_transform, mode='test')
    val_loader = data.DataLoader(val_data, batch_size=5)

    for epoch in range(start_epoch, n_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(desc=f'Epoch {epoch + 1}/{config.n_epochs}')

        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        for images in pbar:
            real_A = images['A'].to(config.device)
            real_B = images['B'].to(config.device)

            fake_A = G_BA(real_B)
            fake_B = G_AB(real_A)

            d_a_fake = D_A(fake_A.detach())
            d_b_fake = D_B(fake_B.detach())

            real = torch.ones_like(d_a_fake)
            fake = torch.zeros_like(d_a_fake)

            loss_r_a = gan_loss(D_A(real_A), real)
            loss_f_a = gan_loss(D_A(fake_A.detach()), fake)
            loss_d_a = (loss_r_a + loss_f_a) / 2

            optim_DA.zero_grad()
            loss_d_a.backward()
            optim_DA.step()

            loss_r_b = gan_loss(D_B(real_B), real)
            loss_f_b = gan_loss(D_B(fake_B.detach()), fake)
            loss_d_b = (loss_r_b + loss_f_b) / 2

            optim_DB.zero_grad()
            loss_d_b.backward()
            optim_DB.step()

            loss_D = (loss_d_a + loss_d_b) / 2

            # gan loss
            d_a_fake = D_A(fake_A)
            d_b_fake = D_B(fake_B)
            loss_g_ab = gan_loss(d_b_fake, real)
            loss_g_ba = gan_loss(d_a_fake, real)
            loss_GAN = (loss_g_ab + loss_g_ba) / 2

            # cyc loss
            loss_cyc_a = cyc_loss(G_BA(fake_B), real_A)
            loss_cyc_b = cyc_loss(G_AB(fake_A), real_B)
            loss_cycle = (loss_cyc_a + loss_cyc_b) / 2

            # Identity loss
            loss_id_a = l_identity(G_BA(real_A), real_A)
            loss_id_b = l_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_a + loss_id_b) / 2

            # Total generators loss
            loss_G = loss_GAN + config.cyc_lambda * loss_cycle + config.id_lambda * loss_identity

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            pbar.set_postfix(
                {'D Loss': '{:.5f}'.format(round(loss_D.item(), 4)),
                 'G Loss': '{:.5f}'.format(round(loss_G.item(), 4))})

        # torch.save(G_AB.state_dict(), f'{weights_folder}/G_AB-checkpoint-{str(epoch).zfill(3)}epoch.pth')
        # torch.save(G_BA.state_dict(), f'{weights_folder}/G_BA-checkpoint-{str(epoch).zfill(3)}epoch.pth')
        test(G_AB, G_BA, val_loader, epoch)
        # TODO: save里保存epoch信息
        # TODO: 增加tensorboard 或 wandb 配置


def test(g_ab, g_ba, val_loader, epoch):
    images = next(iter(val_loader))
    image_a, image_b = images['A'].to(config.device), images['B'].to(config.device)

    g_ab.eval()
    g_ba.eval()

    with torch.no_grad():
        fake_b = g_ab(image_a)
        fake_b = fake_b * 0.5 + 0.5    # 消除正则
        image_a = image_a * 0.5 + 0.5
        output_a2b = torch.cat((image_a, fake_b), 3)
        torchvision.utils.save_image(output_a2b, config.output_dir + f"/output_a2b_{epoch}.png")

        fake_a = g_ba(image_b)
        fake_a = fake_a * 0.5 + 0.5  # 消除正则
        image_b = image_b * 0.5 + 0.5
        output_b2a = torch.cat((image_b, fake_a), 3)
        torchvision.utils.save_image(output_b2a, config.output_dir + f"/output_b2a_{epoch}.png")

    g_ab.train()
    g_ba.train()


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
