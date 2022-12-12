import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.paths = df['path'].values
        self.labels = df['label'].values

        if augmentations is None:
            self.augmentations = transforms.Compose([
                transforms.Resize((180, 180)),
                transforms.ToTensor()
            ])
        else:
            self.augmentations = augmentations

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = self.paths[idx]
        sample = Image.open(sample).convert(mode='RGB')
        sample = self.augmentations(sample)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label

    def __repr__(self):
        return f"Path: {self.paths}, Labels: {self.labels}"

    def __str__(self):
        return f"Path: {self.paths}, Labels: {self.labels}"