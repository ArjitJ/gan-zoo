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
from torch import LongTensor, FloatTensor

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

fmnist = datasets.MNIST(root=".", train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset=fmnist, batch_size=128, shuffle=True
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

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        out = out.view(size, -1)
        source = torch.sigmoid(self.source(out))
        return source, out


class QMI(nn.Module):
    def __init__(self, discete_dim=10, continuous_dim=2):
        super(QMI, self).__init__()
        self.discrete = nn.Linear(64, discete_dim)
        self.continuous = nn.Linear(64, continuous_dim)

    def forward(self, out):
        discrete_code = torch.nn.functional.softmax(self.discrete(out), dim=-1)
        continuous_code = self.continuous(out)
        return discrete_code, continuous_code


criterionSource = nn.BCELoss()
criterionDiscrete = nn.CrossEntropyLoss()
criterionContinuous = nn.MSELoss()
hidden_dim = 100
discrete_dim = 10
continuous_dim = 2
G = Generator(in_channels=hidden_dim, out_channels=1).cuda()
D = Discriminator(in_channels=1, num_labels=10).cuda()
Q = QMI().cuda()
num_classes = 10


def to_categorical(y, num_classes=10):
    return Variable(FloatTensor(np.identity(num_classes)[y]).unsqueeze(-1).unsqueeze(-1)).cuda()


def sample_cont_code(batch_size, size=2):
    return Variable(FloatTensor(batch_size, size, 1, 1).uniform_(-1, 1)).cuda()


optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(itertools.chain(Q.parameters(), G.parameters()), lr=0.001, betas=(0.5, 0.999))
GLosses = []
DLosses = []
BothLosses = []

from torchvision.utils import save_image

static_z = Variable(FloatTensor(torch.randn((num_classes**2, hidden_dim, 1, 1)))).cuda()
static_label = to_categorical(
    np.array([num for _ in range(num_classes) for num in range(num_classes)])
)
static_code = Variable(torch.FloatTensor(num_classes**2, continuous_dim, 1, 1).uniform_(-1, 1)).cuda()


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(torch.randn((n_row**2, hidden_dim, 1, 1)))).cuda()
    static_sample = G(z, static_label, static_code)
    save_image(static_sample.data, "%d.png" % batches_done, nrow=n_row)

    # Get varied c1 and c2
    zeros = np.zeros((n_row**2, continuous_dim-1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1))).unsqueeze(-1).unsqueeze(-1).cuda()
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1))).unsqueeze(-1).unsqueeze(-1).cuda()
    sample1 = G(static_z, static_label, c1)
    sample2 = G(static_z, static_label, c2)
    save_image(sample1.data, "c1%d.png" % batches_done, nrow=n_row)
    save_image(sample2.data, "c2%d.png" % batches_done, nrow=n_row)


for epoch in range(0, 50):
    print(epoch)
    for i, data in enumerate(dataloader):
        trueTensor = torch.Tensor([0.7 + 0.3 * np.random.random()]).cuda()
        falseTensor = torch.Tensor([0.3 * np.random.random()]).cuda()
        images, _ = data
        images = images.cuda()
        batch_size = images.shape[0]
        realSource, _ = D(images)
        realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
        latent = Variable(torch.randn(batch_size, hidden_dim, 1, 1)).cuda()
        random_labels = to_categorical(np.random.randint(0, discrete_dim, batch_size))
        random_cont = sample_cont_code(batch_size)
        fakeData = G(latent, random_labels, random_cont)
        fakeSource, _ = D(fakeData)
        fakeLoss = criterionSource(fakeSource, falseTensor.expand_as(fakeSource))
        lossD = realLoss + fakeLoss
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()

        latent = Variable(torch.randn(batch_size, hidden_dim, 1, 1)).cuda()
        random_labels = to_categorical(np.random.randint(0, discrete_dim, batch_size))
        random_cont = sample_cont_code(batch_size)
        fakeData = G(latent, random_labels, random_cont)
        fakeSource, fakeFeatures = D(fakeData)
        fakeDiscrete, fakeContinuous = Q(fakeFeatures)
        lossG = criterionSource(fakeSource, trueTensor.expand_as(fakeSource))
        lossBoth = criterionDiscrete(fakeDiscrete, random_labels.squeeze(-1).squeeze(-1).argmax(-1).type(torch.long)) \
                   + 0.1 * criterionContinuous(fakeContinuous, random_cont.squeeze(-1).squeeze(-1))
        lossCombined = lossG + lossBoth
        optimizerG.zero_grad()
        lossCombined.backward()
        optimizerG.step()
    print("Loss D {} G {}".format(lossD.cpu(), lossG.cpu()))
    with torch.no_grad():
        sample_image(10, epoch)
    if (epoch + 1) % 5 == 0:
        torch.save(G.state_dict(), "G" + str(epoch) + ".pt")
        torch.save(D.state_dict(), "D" + str(epoch) + ".pt")