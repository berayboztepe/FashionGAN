import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from models.cgan import Generator

path = "FashionMNIST_cGAN/models/generator.pth"
nz = 100
num_classes = 10

model = Generator().to('cpu')
model_state_dict = torch.load(path, map_location='cpu')
model.load_state_dict(model_state_dict)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value)
    ndarr = grid.mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).to('cpu', torch.float).numpy()
    plt.imsave(filename, ndarr)

def generate_images(generator, class_label, num_images=3, latent_dim=100):
    generator.eval()

    noise = torch.randn(num_images, latent_dim, device="cpu")

    labels = torch.full((num_images,), class_label, device="cpu", dtype=torch.long)

    with torch.no_grad():
        generated_images = generator(noise, labels)

    #save_image(generated_images.data, f"images/shirts.png", nrow=5, normalize=True)

    show_tensor_images(generated_images, 3)

for idx in range(10):
    generate_images(model, idx)
