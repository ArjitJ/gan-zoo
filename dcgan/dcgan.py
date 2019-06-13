# -*- coding: utf-8 -*-
import torch
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

mnist = datasets.MNIST(root="data/", train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=512, shuffle=True, num_workers=1
)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.autograd import Variable


def convTBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


def convBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(Generator, self).__init__()
        self.block1 = convTBNReLU(in_channels, 512, 4, 1, 0)
        self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return self.block5(out).tanh()


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        self.in_channels = in_channels
        super(Discriminator, self).__init__()
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return torch.sigmoid(out)


criterion = nn.BCELoss()
G = Generator(in_channels=100, out_channels=1).cuda()
D = Discriminator(in_channels=1).cuda()


optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixedNoise = torch.randn(32, 100, 1, 1).cuda()
img_list = []
GLosses = []
DLosses = []

for epoch in range(51, 100):
    print(epoch)
    for i, data in enumerate(dataloader):
        trueTensor = torch.Tensor([0.7 + 0.3 * np.random.random()]).cuda()
        falseTensor = torch.Tensor([0.3 * np.random.random()]).cuda()
        data = data[0].cuda()
        D.zero_grad()
        realPred = D(data)
        realLoss = criterion(realPred, trueTensor.expand_as(realPred))
        realLoss.backward()
        latent = Variable(torch.randn(data.shape[0], 100, 1, 1)).cuda()
        fakeData = G(latent)
        fakePred = D(fakeData.detach())
        fakeLoss = criterion(fakePred, falseTensor.expand_as(fakePred))
        fakeLoss.backward()
        lossD = realLoss + fakeLoss
        optimizerD.step()
        G.zero_grad()
        lossG = criterion(D(fakeData), trueTensor.expand_as(fakePred))
        lossG.backward()
        optimizerG.step()
        GLosses.append(lossG.cpu())
        DLosses.append(lossD.cpu())
    #         print("Loss D {} G {}".format(lossD.cpu(), lossG.cpu()))
    if (epoch + 1) % 10 == 0:
        torch.save(G.state_dict(), "G" + str(epoch) + ".pt")
        torch.save(D.state_dict(), "D" + str(epoch) + ".pt")
    with torch.no_grad():
        fake = G(fixedNoise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

# G.load_state_dict(torch.load('G99.pt'))
# G.eval()
# D.load_state_dict(torch.load('D99.pt'))
# D.eval()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

plt.plot(GLosses)
plt.plot(DLosses)
plt.show()
