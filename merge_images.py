import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_images_from_folder(folder, num_images, new_size):
    # Load all images first
    all_images = []
    filenames = sorted(os.listdir(folder), reverse=True)[:num_images]
    for filename in filenames:
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            all_images.append(img)

    reordered_images = []
    for start_index in range(3):
        for i in range(start_index, len(all_images), 3):
            reordered_images.append(all_images[i])

    return reordered_images

def create_image_table(images, rows, cols):
    assert len(images) == rows * cols, "Number of images does not match the grid size"

    img_width, img_height = images[0].size
    grid_img = Image.new('RGB', (cols * img_width, rows * img_height))

    for index, image in enumerate(images):
        x = index % cols * img_width
        y = index // cols * img_height
        grid_img.paste(image, (x, y))

    return grid_img

folder = 'images/'
num_images = 30
new_image_size = (600, 600)
rows, cols = 3, 10

images = load_images_from_folder(folder, num_images, new_image_size)

table_image = create_image_table(images, rows, cols)

plt.figure(figsize=(12, 12))
plt.imshow(table_image)
plt.axis('on')

x_positions = np.linspace(start=0, stop=table_image.width, num=cols, endpoint=False)
x_positions += (x_positions[1] - x_positions[0]) / 2 

y_positions = np.linspace(start=0, stop=table_image.height, num=rows, endpoint=False)
y_positions += (y_positions[1] - y_positions[0]) / 2

class_names = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

y_names = ["Real images", "GAN images", "CGAN images"]

plt.xticks(ticks=x_positions, labels=sorted(class_names, reverse=True))
plt.yticks(ticks=y_positions, labels=y_names)
plt.show()
