import os
import struct
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class FashionMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = self.read_images(images_path)
        self.labels = self.read_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def read_images(filepath):
        with open(filepath, 'rb') as f:
            _, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
        return images

    @staticmethod
    def read_labels(filepath):
        with open(filepath, 'rb') as f:
            _, num = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

def get_fashion_mnist_data(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    root_folder = "../../Issue_Generate_GANS_forAll_Classes/fashion_mnist/"
    train_set = torchvision.datasets.FashionMNIST(
        root = root_folder,
        train = True,
        download = True,
        transform=transform
        )

    

    #dataset = FashionMNISTDataset(images_path, labels_path, transform=transform)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=train)

    return dataloader
