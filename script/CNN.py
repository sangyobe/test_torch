import torch
from torch import nn

class CNN(nn.Module):
  def __init__(self, channel, width, height, classes):
    super().__init__()

    self.conv1 = nn.Sequential(nn.Conv2d(channel, 10, 3, padding=1),
                               nn.BatchNorm2d(10),
                               nn.ReLU())
    self.maxpool1 = nn.MaxPool2d(2); width //= 2; height //= 2
    self.conv2 = nn.Sequential(nn.Conv2d(10, 20, 3, padding=1),
                               nn.BatchNorm2d(20),
                               nn.ReLU())
    self.maxpool2 = nn.MaxPool2d(2); width //= 2; height //= 2
    self.conv3 = nn.Sequential(nn.Conv2d(20, 40, 3, padding=1),
                               nn.BatchNorm2d(40),
                               nn.ReLU())
    self.maxpool3 = nn.MaxPool2d(2); width //= 2; height //= 2
    self.fc1 = nn.Linear(40*width*height, 100)
    self.fc2 = nn.Linear(100, classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.conv3(x)
    x = self.maxpool3(x)
    x = x.view(x.shape[0], -1) # (batch size, ) 의 2차원 tensor로 변환
    # x = x.flatten(dim=1) # batch size 만 남기고 flatten()
    x = self.fc1(x)
    x = self.fc2(x)
    return x

class CNN_deep(nn.Module):
  def __init__(self, channel, width, height, classes):
    super().__init__()

    self.conv1 = nn.Sequential(nn.Conv2d(channel, 32, 3, padding=1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),
                               nn.BatchNorm2d(32),
                               nn.ReLU())
    self.maxpool1 = nn.MaxPool2d(2); width //= 2; height //= 2
    self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU())
    self.maxpool2 = nn.MaxPool2d(2); width //= 2; height //= 2
    self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.Conv2d(128, 128, 3, padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU())
    self.maxpool3 = nn.MaxPool2d(2); width //= 2; height //= 2
    self.fc = nn.Sequential(nn.Linear(128*width*height, 512),
                            nn.Linear(512, classes))

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.conv3(x)
    x = self.maxpool3(x)
    x = x.view(x.shape[0], -1) # (batch size, ) 의 2차원 tensor로 변환
    # x = x.flatten(dim=1) # batch size 만 남기고 flatten()
    x = self.fc(x)
    return x