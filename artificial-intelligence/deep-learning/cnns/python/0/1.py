import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_worker=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# output: Files already downloaded and verified

def imshow(img):
    plt.figure(figsize=(10, 10))
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imgshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))


class ConvNet(nn.Module):

    def __init__(self, n_channels, n_output_neurons):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True)
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True)
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, n_output_neurons)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.main(x)
        # x shape [batch_size, 64, 1, 1]
        x = x.view(batch_size, -1)
        return self.mlp(x)


height, width = images.shape[2:]
net = ConvNet(n_channels=3, n_output_neurons=10).cuda()
net(images.cuda()).shape
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


def get_quality(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))

get_quality(net)
