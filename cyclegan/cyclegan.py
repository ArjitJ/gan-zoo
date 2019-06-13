# !wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
# !unzip horse2zebra.zip
# !mv horse2zebra/horse2zebra horse2zebra

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import itertools


class cycleDataset(Dataset):
    def __init__(self, path):
        super(cycleDataset, self).__init__()
        self.filesA = []
        for i in os.listdir(path + "A"):
            if i.endswith(".jpg"):
                self.filesA.append(path + "A/" + i)
        self.filesB = []
        for i in os.listdir(path + "B"):
            if i.endswith(".jpg"):
                self.filesB.append(path + "B/" + i)
        self.sizeA = len(self.filesA)  # get the size of dataset A
        self.sizeB = len(self.filesB)  # get the size of dataset B

    def __getitem__(self, idx):
        fileA = self.filesA[idx % self.sizeA]  # make sure index is within then range
        fileB = self.filesB[idx % self.sizeB]
        imgA = transforms.ToTensor()(Image.open(fileA).convert("RGB"))
        imgB = transforms.ToTensor()(Image.open(fileB).convert("RGB"))
        return imgA, imgB

    def __len__(self):
        return max(self.sizeA, self.sizeB)


class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels=in_channels, out_channels=64, kernel_size=7, padding=0
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.res = nn.Sequential(
            residual_block(256),
            residual_block(256),
            residual_block(256),
            residual_block(256),
            residual_block(256),
            residual_block(256),
            residual_block(256),
            residual_block(256),
            residual_block(256),
        )
        self.decon1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.decon2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res(out)
        out = self.decon1(out)
        out = self.decon2(out)
        return self.conv4(out)


class discriminator(nn.Module):
    def __init__(self, in_channels):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.conv4(out)


in_channels = 3
out_channels = 3
GXtoY = generator(in_channels=in_channels, out_channels=out_channels).cuda()
GYtoX = generator(in_channels=out_channels, out_channels=in_channels).cuda()
DX = discriminator(in_channels=out_channels).cuda()
DY = discriminator(in_channels=in_channels).cuda()
criterionGAN = nn.MSELoss().cuda()
criterionCycle = nn.L1Loss().cuda()
optimizer_G = torch.optim.Adam(
    itertools.chain(GXtoY.parameters(), GYtoX.parameters()),
    lr=0.0002,
    betas=(0.5, 0.999),
)
optimizer_D = torch.optim.Adam(
    itertools.chain(DX.parameters(), DY.parameters()), lr=0.0002, betas=(0.5, 0.999)
)
trueTensor = torch.Tensor([1.0]).cuda()
falseTensor = torch.Tensor([0.0]).cuda()
lambdaCycle = 10

path = "./horse2zebra/train"
dataloader = DataLoader(cycleDataset(path), batch_size=1)
import itertools

for epoch in range(0, 100):
    print(epoch)
    for idx, data in enumerate(dataloader):
        realX = data[0].cuda()
        realY = data[1].cuda()
        fakeX = GYtoX(realY)
        fakeY = GXtoY(realX)
        reconstructedX = GYtoX(fakeY)
        reconstructedY = GXtoY(fakeX)
        realPredX = DX(realX)
        realPredY = DY(realY)
        fakePredX = DX(fakeX)
        fakePredY = DY(fakeY)
        #         print(realX.shape, realY.shape, reconstructedX.shape, reconstructedY.shape)
        lossDX = (
            criterionGAN(realPredX, trueTensor.expand_as(realPredX))
            + criterionGAN(fakePredX, falseTensor.expand_as(fakePredX))
        ) / 2
        lossDY = (
            criterionGAN(realPredY, trueTensor.expand_as(realPredY))
            + criterionGAN(fakePredY, falseTensor.expand_as(fakePredY))
        ) / 2
        lossAdvGXtoY = criterionGAN(fakePredY, trueTensor.expand_as(fakePredY))
        lossAdvGYtoX = criterionGAN(fakePredX, trueTensor.expand_as(fakePredX))
        lossCycleGX = criterionCycle(reconstructedX, realX)
        lossCycleGY = criterionCycle(reconstructedY, realY)
        lossG = lossAdvGXtoY + lossAdvGYtoX + lambdaCycle * (lossCycleGX + lossCycleGY)
        for i in DX.parameters():
            i.requires_grad = False
        for i in DY.parameters():
            i.requires_grad = False
        optimizer_G.zero_grad()
        lossG.backward(retain_graph=True)
        optimizer_G.step()
        for i in DX.parameters():
            i.requires_grad = True
        for i in DY.parameters():
            i.requires_grad = True
        optimizer_D.zero_grad()
        lossDX.backward(retain_graph=True)
        fakeX.detach()
        lossDY.backward(retain_graph=True)
        fakeY.detach()
        optimizer_D.step()
    #         print(idx, lossDY, lossDY, lossG)
    torch.save(GXtoY.state_dict(), "GXtoY" + str(epoch) + ".pt")
    torch.save(GYtoX.state_dict(), "GYtoX" + str(epoch) + ".pt")
    torch.save(DX.state_dict(), "DX" + str(epoch) + ".pt")
    torch.save(DY.state_dict(), "DY" + str(epoch) + ".pt")

GXtoY.load_state_dict(torch.load("GXtoY99.pt"))
GXtoY.eval()
GYtoX.load_state_dict(torch.load("GYtoX99.pt"))
GYtoX.eval()
DX.load_state_dict(torch.load("DX99.pt"))
DX.eval()
DY.load_state_dict(torch.load("DY99.pt"))
DY.eval()

path = "./horse2zebra/test"
dataloader = DataLoader(cycleDataset(path), batch_size=1)

import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
for iter, i in enumerate(dataloader):
    inpA = i[0].cuda()
    inpB = i[1].cuda()
    out1 = GXtoY(inpA)
    out2 = GYtoX(inpB)
    rec2 = GXtoY(out2)
    rec1 = GYtoX(out1)
    inpA = inpA.cpu().detach().numpy()[0, :, :, :].T
    inpB = inpB.cpu().detach().numpy()[0, :, :, :].T
    out1 = out1.cpu().detach().numpy()[0, :, :, :].T
    out2 = out2.cpu().detach().numpy()[0, :, :, :].T
    rec1 = rec1.cpu().detach().numpy()[0, :, :, :].T
    rec2 = rec2.cpu().detach().numpy()[0, :, :, :].T
    f1, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    ax1.imshow(inpA)
    ax2.imshow(out1)
    ax3.imshow(rec1)
    plt.savefig("results/" + "A" + str(iter) + ".png")
    plt.close()
    f2, ((ax4, ax5, ax6)) = plt.subplots(1, 3)
    ax4.imshow(inpB)
    ax5.imshow(out2)
    ax6.imshow(rec2)
    plt.savefig("results/" + "B" + str(iter) + ".png")
    plt.close()
