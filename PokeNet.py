import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
"""
Creating the dataset and the dataloader. 
Remember to override the __getitem__ and __len__

"""
trainset_path ='/home/mcv/datasets/MIT_split/train/'
testset_path ='/home/mcv/datasets/MIT_split/test/'

trainset = torchvision.datasets.ImageFolder(root=trainset_path , transform = transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)



trainset = torchvision.datasets.ImageFolder(root=testset_path, transform = transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


"""

Creating the PokeNet

"""

class PokeNet(nn.Module):
    def __init__(self):
        super(PokeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.BN1 = nn.BatchNorm2d()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(3, 6, 5)
        self.BN2 = nn.BatchNorm2d()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(3, 6, 5)
        self.BN3 = nn.BatchNorm2d()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.Drop = nn.Dropout2d() #0.5 by default

        self.conv4 = nn.Conv2d(3, 6, 5)
        self.BN4 = nn.BatchNorm2d()

        self.pool4 = nn.AvgPool2d(2, 2)


    def forward(self, x):
        x = self.pool1(F.relu(self.BN1(self.conv1(x))))
        x = self.pool2(F.relu(self.BN2(self.conv2(x))))
        x = self.Drop(self.pool3(F.relu(self.BN3(self.conv3(x)))))
        x = F.relu(self.BN4(self.conv4(x)))
        x = self.pool4(x)
        x = F.softmax(x)
        return x


net = PokeNet()

"""

Set up an optimizer and define a loss function

"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


"""

Train the model

"""
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))