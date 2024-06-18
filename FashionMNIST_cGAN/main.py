import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torchvision.utils import save_image
from models.cgan import Generator, Discriminator
from data.download_data import get_fashion_mnist_data
from utils.training import train_cgan
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

wandb.init(project="FashionMNIST_cGAN")

epochs = 100
batch_size = 64
lr = 0.0002
nz = 100
num_classes = 10

train_cgan(epochs, batch_size, lr, nz, num_classes)

test_loader = get_fashion_mnist_data(batch_size, train=False)

real_images, _ = next(iter(test_loader))
real_images = real_images.to('cpu')

generator = Generator(nz, 784, 256, num_classes).to('cpu')
generator.eval()

noise = torch.randn(batch_size, nz, device='cpu')
labels = torch.randint(0, num_classes, (batch_size,), device='cpu')
generated_images = generator(noise, labels)

#show_tensor_images(generated_images)
