import torch
from torch import nn

class MLP(nn.Module):
  def __init__(self, channel, width, height, classes):
    super().__init__()

    self.fcs = nn.Sequential(nn.Linear(channel * width * height, 100),
                             nn.ReLU(),
                             nn.Linear(100, classes))

  def forward(self, x):
    x = x.view(x.shape[0], -1) #
    # x = x.flatten(dim=1) # batch size 만 남기고 flatten()
    x = self.fcs(x)
    return x

class MLP_shallow(nn.Module):
  def __init__(self, channel, width, height, classes):
    super().__init__()

    self.fcs = nn.Sequential(nn.Linear(channel * width * height, 100),
                             nn.BatchNormal1d(100),
                             nn.ReLU(),
                             nn.Linear(100, classes))

  def forward(self, x):
    x = x.view(x.shape[0], -1) #
    # x = x.flatten(dim=1) # batch size 만 남기고 flatten()
    x = self.fcs(x)
    return x

class MLP_deep(nn.Module):
  def __init__(self, channel, width, height, classes):
    super().__init__()

    self.fcs = nn.Sequential(nn.Linear(channel * width * height, 75),
                             nn.BatchNormal1d(75),
                             nn.ReLU(),
                             *[i for _ in range(13) for i in [nn.Linear(75, 75), nn.ReLU()]],
                             nn.Linear(75, classes))

  def forward(self, x):
    x = x.view(x.shape[0], -1) #
    # x = x.flatten(dim=1) # batch size 만 남기고 flatten()
    x = self.fcs(x)
    return x