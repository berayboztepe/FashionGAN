import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from models.cgan import Generator, Discriminator
from data.download_data import get_fashion_mnist_data
import os

def train_cgan(epochs, batch_size, lr, nz, num_classes):
    train_loader = get_fashion_mnist_data(batch_size)
    
    generator = Generator(nz, 784, 256, num_classes).to('cpu')
    discriminator = Discriminator(784, 256, num_classes).to('cpu')
    
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    
    wandb.init(project="FashionMNIST_cGAN")

    for epoch in range(epochs):
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1).to('cpu')
            fake = torch.zeros(batch_size, 1).to('cpu')

            real_imgs = imgs.view(batch_size, -1).to('cpu')
            labels = labels.to('cpu')

            # Train Generator
            optimizer_G.zero_grad()
            
            noise = torch.randn(batch_size, nz, device='cpu')
            gen_labels = torch.randint(0, num_classes, (batch_size,), device='cpu')
            gen_imgs = generator(noise, gen_labels)
            
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = criterion(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            real_validity = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_validity, valid)
            
            fake_validity = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = criterion(fake_validity, fake)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            mean_generator_loss += g_loss.item()
            mean_discriminator_loss += d_loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(train_loader)} \
                      Loss D: {d_loss.item()}, loss G: {g_loss.item()}")
        
        mean_generator_loss /= len(train_loader)
        mean_discriminator_loss /= len(train_loader)
        
        wandb.log({"mean_generator_loss": mean_generator_loss, "mean_discriminator_loss": mean_discriminator_loss})

    model_path = os.path.join("FashionMNIST_cGAN/models", "cgan_checkpoint.pth")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_optimizer_state_dict': optimizer_G.state_dict(),
        'disc_optimizer_state_dict': optimizer_D.state_dict(),
        'generator_loss': g_loss.item(),
        'discriminator_loss': d_loss.item(),
    }, model_path)