import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(nz + num_classes, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, nc),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        input = torch.cat((noise, labels), -1)
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(nc + num_classes, ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = self.label_embedding(labels)
        input = torch.cat((img, labels), -1)
        return self.model(input)
