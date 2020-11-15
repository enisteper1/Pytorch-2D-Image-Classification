import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self,nc, img_size = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        x = torch.randn(img_size, img_size).view(-1, 1, img_size, img_size)
        self.linearized = None
        self.convs(x)

        self.fc1 = torch.nn.Linear(in_features=self.linearized, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=nc)

    def convs(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), (2, 2))

        if self.linearized is None:
            self.linearized = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.linearized)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x