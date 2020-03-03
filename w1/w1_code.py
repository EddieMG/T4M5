import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
now = datetime.now()
date_time = now.strftime("pol_%m-%d-%Y_%H:%M:%S")
print("---------------------------------------------------------")
print("                   " + date_time + "                     ")
print("---------------------------------------------------------")

gpu_id = ""
torch.cuda.set_device(device=gpu_id)

train_data_dir='/home/mcv/datasets/MIT_split/train'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 256
img_height = 256
batch_size  = 4
epochs = 30
validation_samples = 807

# TO DO: Si s'executa en GPU s'ha d'especificar (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

# -----------------------------------------------------------------------------------
#                            Loading and normalizing data
# -----------------------------------------------------------------------------------
# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root=train_data_dir,
                                            transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.ImageFolder(root=test_data_dir,
                                            transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

print("trainloader")
print(trainloader)

def imsave(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imsave(date_time+".png", np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


# show images
imsave(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# -----------------------------------------------------------------------------------
#                            Neural Network
# -----------------------------------------------------------------------------------

'''class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = torch.nn.Softmax

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.softmax(output)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

net = Net()
# -----------------------------------------------------------------------------------
#                            Loss function and optimizer
# -----------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# -----------------------------------------------------------------------------------
#                                Train the network
# -----------------------------------------------------------------------------------
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
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

# -----------------------------------------------------------------------------------
#                                Save the model
# -----------------------------------------------------------------------------------
os.mkdir(date_time)
PATH = '.'+date_time+'/model.pth'
torch.save(net.state_dict(), PATH)

# -----------------------------------------------------------------------------------
#                             Test the network (all classes)
# -----------------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------------
#                     Test the network (classes individually)
# -----------------------------------------------------------------------------------
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))'''


