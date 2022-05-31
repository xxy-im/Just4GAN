import os
import glob

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.fnames = glob.glob(os.path.join(root_dir, '*'))
        self.transform = transform
        self.len = len(self.fnames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = idx % self.len
        img = torchvision.io.read_image(self.fnames[i])
        input_image = img[:, :, 512:]
        target_image = img[:, :, :512]
        return self.transform(input_image), self.transform(target_image)


if __name__ == "__main__":
    train_transform = [
        # transforms.Resize((256, 256)),
        transforms.ToPILImage(),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    train_transform = transforms.Compose(train_transform)

    dataset = AnimeDataset("../../data/Anime Sketch Colorization Pair/data/train", train_transform)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        break

