import os
import glob
import config

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


class Paint2PhotoDataset(Dataset):
    def __init__(self, root_dir, transform, mode='train'):
        self.files_A = glob.glob(os.path.join(root_dir, "%sA" % mode) + "/*.*")
        self.files_B = glob.glob(os.path.join(root_dir, "%sB" % mode) + "/*.*")
        self.transform = transform
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        i = idx % self.len_A
        j = idx % self.len_B

        img_A = Image.open(self.files_A[i])
        img_B = Image.open(self.files_B[j])

        return {"A": self.transform(img_A), "B": self.transform(img_B)}


if __name__ == "__main__":
    train_transform = [
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    train_transform = transforms.Compose(train_transform)

    dataset = Paint2PhotoDataset(config.data_dir, train_transform)
    loader = DataLoader(dataset, batch_size=5)
    for images in loader:
        torchvision.utils.save_image(images['A'], f"./x.png")
        torchvision.utils.save_image(images['B'], f"./y.png")
        break

