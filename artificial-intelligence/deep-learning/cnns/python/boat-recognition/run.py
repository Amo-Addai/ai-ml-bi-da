import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize([224, 224]),

    # sample data augmentations
    transforms.RandomRotation(degree=[-30, 30]), # rotation data augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder('./data/', transform=transform)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

print(len(datasets), dataset.classes)

def imshow(img):
    plt.figure(figsize=(10, 10))
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))


# 1st, create the neural network
net = torchvision.models.resnet18(pretrained=True)
print(net)

for param in net.parameters():
    param.requires_grad = False

net.fc = nn.Linear(512, 9)
print(net)

net = net.cuda()


net(images.cuda()).shape

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


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
    
    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))


get_quality(net)


def train_network(net, n_epochs=10):
    losses = []

    for epoch in range(n_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
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
            losses.append(loss.item())
        
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

        get_quality(net)
    
    plt.plot(losses, 'b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    print('Finished Training')


train_network(net, n_epochs=5)
