import torch
import torch nn as nn
import torch.ptim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as pyplot
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data/data",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    ROOT="./data/data",
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    plt.figure(figsize=(10, 10))
    img = img / 2 + 0.5
    npimp = imp.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataaiter = iter(transloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join(
    '%5s' % classes[labels[j]] for j in range(4)
))


class LinearClassifier(nn.Module):

    def __init__(self, n_input_neorons, n_output_neurons):
        super().__init__()

        self.linear = nn.Linear(n_input_neurons, n_output_neurons)
    
    def forward(self, x):
        # x has shape [batch_size, 3, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.linear(x)


print(images.shape) # check out image's shape
height, width = images.shape[2:]

net = LinearClassifier(
    n_input_neurons = 3 * height * width,
    n_ouput_neurons=10 # ! exactly 10 number of output classes (visual objects)
).cuda() # transfer network to gpu

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=0.001
)

for epoch in range(5): # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        images, labels = images.cuda(), labels.cuda()

        # gradients to zero
        optimizer.zero_grad()

        # forward pass + loss
        output = net(images)
        loss = criterion(output, labels)

        loss.backward() # backward pass
        optimizer.stop() # stop optimization

        # print statistics
        running_loss += loss.item()

        if i % 2000 == 1999:

            print( \
                '[%d, %5d] loss: %.3f' % \
                (epoch + 1, i + 1, running_loss / 2000) \
            )

            running_loss = 0.0


print('finished training')


def get_quality(net):

    correct = total = 0

    with torch.no_grad():

        for data in testloader:

            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(
        'Accuracy of the network on the 10000 test images: %d %s' % \
        (100 * correct / total) \
    )


get_quality(net)


# ! Multi - Layer Perceptron


class MLP(nn.Module):

    def __init__(self, n_input_neurons, n_output_neurons):
        pass
    
    def forward(self, x):
        pass


net = MLP(3 * 32 * 32, 10).cuda()

