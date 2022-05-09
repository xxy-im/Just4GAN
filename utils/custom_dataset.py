import os
from PIL import Image

import torch.utils.data as data


def read_file(filename):
    image_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip().split(' ')
            name = content[0]
            image_list.append(name)
        return image_list


class TrainDataset(data.Dataset):
    def __init__(self, filename, image_dir):
        self.image_list = read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_list)

    def __getitem__(self, index):
        i = index % self.len
        image_name = self.image_list[i]
        image_path = os.path.join(self.image_dir, image_name)
        img = Image.open(image_path)
        return img

    def __len__(self):
        return self.len

