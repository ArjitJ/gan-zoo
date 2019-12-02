# -*- coding: utf-8 -*-
from torchvision import transforms
import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import LongTensor

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

fmnist = datasets.FashionMNIST(root="data/", train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset=fmnist, batch_size=32, shuffle=True, num_workers=1
)


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
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = convTBNReLU(in_channels + 10 + 2, 512, 4, 1, 0)
        self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    def forward(self, noise, discrete_code, continuous_code):
        input = torch.cat((noise, discrete_code, continuous_code), 1)
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return self.block5(out).tanh()


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_labels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv2d(512, 64, 4, 1, 0)
        self.source = nn.Linear(64, 1)
        self.discrete = nn.Linear(64, 10)
        self.continuous = nn.Linear(64, 2)

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        source = torch.sigmoid(self.source(out.view(size, -1)))
        discrete_code = torch.nn.functional.softmax(self.discrete(out.view(size, -1)), dim=-1)
        continuous_code = self.continuous(out.view(size, -1))
        return source, discrete_code, continuous_code


criterionSource = nn.BCELoss()
criterionDiscrete = nn.CrossEntropyLoss()
criterionContinuous = nn.MSELoss()
G = Generator(in_channels=100, out_channels=1).cuda()
D = Discriminator(in_channels=1, num_labels=10).cuda()
num_classes = 10
def to_categorical(y, num_classes=10):
    y_new = torch.zeros((y.shape[0], num_classes, 1, 1))
    y_new[:, y, :, :] = 1
    return Variable(y_new).cuda()

def sample_cont_code(size=2):
    return Variable(torch.FloatTensor(32, size, 1, 1).uniform_(-1, 1)).cuda() 

optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerBoth = optim.Adam(itertools.chain(D.parameters(), G.parameters()), lr=0.0002, betas=(0.5, 0.999))
fixedNoise = torch.randn(32, 100, 1, 1).cuda()
fixedLabels = to_categorical(np.random.randint(0, 10, 32))
fixedCont = sample_cont_code()
img_list = []
GLosses = []
DLosses = []
BothLosses = []

for epoch in range(0, 50):
    print(epoch)
    for i, data in enumerate(dataloader):
        trueTensor = torch.Tensor([0.7 + 0.3 * np.random.random()]).cuda()
        falseTensor = torch.Tensor([0.3 * np.random.random()]).cuda()
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        
        D.zero_grad()
        realSource, _, _ = D(images)
        realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
        realLoss.backward()
        latent = Variable(torch.randn(images.shape[0], 100, 1, 1)).cuda()
        random_labels = to_categorical(np.random.randint(0, 10, images.shape[0]))
        random_cont = sample_cont_code()
        fakeData = G(latent, random_labels, random_cont)
        fakeSource, _, _ = D(fakeData.detach())
        fakeLoss = criterionSource(fakeSource, falseTensor.expand_as(fakeSource))
        fakeLoss.backward()
        lossD = realLoss + fakeLoss
        optimizerD.step()
        
        G.zero_grad()
        fakeSource, _, _ = D(fakeData)
        lossG = criterionSource(fakeSource, trueTensor.expand_as(fakeSource))
        lossG.backward()
        optimizerG.step()
        
        optimizerBoth.zero_grad()
        latent = Variable(torch.randn(images.shape[0], 100, 1, 1)).cuda()
        random_labels = Variable(torch.Tensor(np.random.randint(0, 10, images.shape[0])).type(torch.long)).cuda()
        random_cont = sample_cont_code()
        fakeData = G(latent, to_categorical(random_labels), random_cont)
        _, fakeDiscrete, fakeContinuous = D(fakeData)
        lossBoth = criterionDiscrete(fakeDiscrete, random_labels) \
                + 0.1*criterionContinuous(fakeContinuous, random_cont.squeeze(-1).squeeze(-1))
        lossBoth.backward()
        optimizerBoth.step()
        GLosses.append(lossG.cpu())
        DLosses.append(lossD.cpu())
        BothLosses.append(lossBoth.cpu())
    if (epoch + 1) % 10 == 0:
        print("Loss D {} G {}".format(lossD.cpu(), lossG.cpu()))
        torch.save(G.state_dict(), "G" + str(epoch) + ".pt")
        torch.save(D.state_dict(), "D" + str(epoch) + ".pt")
    with torch.no_grad():
        fake = G(fixedNoise, fixedLabels, fixedCont).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

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