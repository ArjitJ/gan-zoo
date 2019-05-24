import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
class cycleDataset(Dataset):
    def __init__(self, path):
        super(cycleDataset, self).__init__()
        self.filesA = []
        for i in os.listdir(path+'A'):
            if i.endswith('.jpg'):
                self.filesA.append(path + 'A/' + i)
        self.filesB = []
        for i in os.listdir(path + 'B'):
            if i.endswith('.jpg'):
                self.filesB.append(path + 'B/' + i)
        self.sizeA = len(self.filesA)  # get the size of dataset A
        self.sizeB = len(self.filesB)  # get the size of dataset B

    def __getitem__(self, idx):
        fileA = self.filesA[idx % self.sizeA]  # make sure index is within then range
        fileB = self.filesB[idx % self.sizeB]
        imgA = transforms.ToTensor()(Image.open(fileA).convert('RGB'))
        imgB = transforms.ToTensor()(Image.open(fileB).convert('RGB'))
        return imgA, imgB

    def __len__(self):
        return max(self.sizeA, self.sizeB)

class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, x):
        return x+self.block(x)

class generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
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
            residual_block(256)
        )
        self.decon1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.decon2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0),
            nn.Tanh())

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
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
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
path = './horse2zebra/train'
dataloader = DataLoader(cycleDataset(path), batch_size=1)
import itertools
lambdaCycle = 10
num_epochs = 5
criterionGAN = nn.MSELoss().cuda()
criterionCycle = nn.L1Loss().cuda()
optimizer_G = torch.optim.Adam(itertools.chain(GXtoY.parameters(), GYtoX.parameters()), lr=0.0002,
                                    betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(DX.parameters(), DY.parameters()), lr=0.0002,
                                    betas=(0.5, 0.999))
trueTensor = torch.Tensor([1.0]).cuda()
falseTensor = torch.Tensor([0.0]).cuda()
for epoch in range(num_epochs):
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
        lossDX = (criterionGAN(realPredX, trueTensor.expand_as(realPredX)) + criterionGAN(fakePredX, falseTensor.expand_as(fakePredX)))/2
        lossDY = (criterionGAN(realPredY, trueTensor.expand_as(realPredY)) + criterionGAN(fakePredY, falseTensor.expand_as(fakePredY)))/2
        lossAdvGXtoY = criterionGAN(fakePredY, trueTensor.expand_as(fakePredY))
        lossAdvGYtoX = criterionGAN(fakePredX, trueTensor.expand_as(fakePredX))
        lossCycleGX = criterionCycle(reconstructedX, realX)
        lossCycleGY = criterionCycle(reconstructedY, realY)
        lossG = lossAdvGXtoY + lossAdvGYtoX + lambdaCycle*(lossCycleGX + lossCycleGY)
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
        lossDY.backward(retain_graph=True)
        optimizer_D.step()
        print(idx, lossDY, lossDY, lossG)