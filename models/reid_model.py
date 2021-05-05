import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(torch.nn.Module):
    def __init__(self):
      super(LeNet, self).__init__()
      
      # input channel = 1, output channels = 6, kernel size = 5
      # input image size = (28, 28), image output size = (24, 24)
      self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
      
      # input channel = 6, output channels = 16, kernel size = 5
      # input image size = (12, 12), output image size = (8, 8)
      self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
      
      # input dim = 4 * 4 * 16 ( H x W x C), output dim = 120
      self.fc3 = torch.nn.Linear(in_features=4 * 4 * 16, out_features=120)
      
      # input dim = 120, output dim = 84
      self.fc4 = torch.nn.Linear(in_features=120, out_features=84)
      
      # input dim = 84, output dim = 10
      self.fc5 = torch.nn.Linear(in_features=84, out_features=10)
      
    def forward(self, x):
      
      x = self.conv1(x)
      x = F.relu(x)
      # Max Pooling with kernel size = 2
      # output size = (12, 12)
      x = F.max_pool2d(x, kernel_size=2)
      
      x = self.conv2(x)
      x = F.relu(x)
      # Max Pooling with kernel size = 2
      # output size = (4, 4)
      x = F.max_pool2d(x, kernel_size=2)
      
      # flatten the feature maps into a long vector
      x = x.view(x.shape[0], -1)
      #(bs, 4*4*16)
      x = self.fc3(x)
      x = F.relu(x)
      
      x = self.fc4(x)
      x = F.relu(x)
      
      x = self.fc5(x)

      return x
