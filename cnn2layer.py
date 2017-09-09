import torch
import torch.nn as nn
from torch.autograd import Variable


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, kernel_size1=5, dropout1=0.5, stride1=1, kernel_size2=5, stride2=1,
                 dropout2=0.1, padding1=2, pool_kernel1=2, padding2=2, pool_kernel2=2):
        super(CNN, self).__init__()
        self.kernel_size1 = kernel_size1
        self.stride1 = stride1
        self.padding1 = padding1
        self.pool_kernel1 = pool_kernel1
        self.dropout1 = dropout1
        self.kernel_size2 = kernel_size2
        self.stride2 = stride2
        self.padding2 = padding2
        self.pool_kernel2 = pool_kernel2
        self.dropout2 = dropout2

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1),
            #nn.BatchNorm2d(6),
            nn.Dropout(p=self.dropout1),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_kernel1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2),
            #nn.BatchNorm2d(16),
            nn.Dropout(p=self.dropout2),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_kernel2))
        self.fc = nn.Linear(12 * 8 * 8, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, out.size(0))
        out = self.fc(out)
        return out
