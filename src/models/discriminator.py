from torch import nn
import torch

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1,28,28)):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + num_classes, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img, labels):
        labels = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.expand(labels.size(0), labels.size(1), img.size(2), img.size(3))
        d_in = torch.cat((img, labels), 1)
        validity = self.model(d_in)
        return validity.view(-1, 1).sigmoid()