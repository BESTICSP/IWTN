path_real = '/home/user/disk_3/xzx/F-LSeSim/results/street_selfs2n/test_300/images/real_A/'
path_fake = '/home/user/disk_3/xzx/F-LSeSim/results/street_multiSEstc/test_240/images/fake_B'

import torch
#_ = torch.manual_seed(123)
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

inception = InceptionScore()

dataset = ImageDataset(path_fake, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

for batch in dataloader:
    imgs = batch.type(torch.uint8)
    inception.update(imgs)

print(inception.compute())

