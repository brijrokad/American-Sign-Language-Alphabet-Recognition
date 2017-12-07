import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

epochs = 5
batch_size = 100
lr = 0.001

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_data = dsets.MNIST(root='./data',
                        train=False,
                        transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class CNN(nn.Module):

    def __init__(self):
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=True),
            nn.Relu(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=True),
            nn.Relu(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
    	out = self.layer1(x)
    	out = self.layer2(out)
    	out = self.view(out.size(0), -1)
