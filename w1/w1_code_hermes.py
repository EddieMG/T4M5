import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

trainset_path ='/home/mcv/datasets/MIT_split/train/'
testset_path ='/home/mcv/datasets/MIT_split/test/'

batch_size = 4

trans = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.ImageFolder(root=trainset_path , transform=trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=2)


testset = torchvision.datasets.ImageFolder(root=testset_path, transform=trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']


# functions to show an image

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels



class PolNet(nn.Module):
    def __init__(self):
        super(PolNet, self).__init__()
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


net = PolNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def val_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


data_loaders = {"train": trainloader, "val": testloader}
data_lengths = {"train": len(trainloader), "val": len(testloader)}


def train_model(optimizer, criterion, epochs=20):
    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in data_loaders[phase]:
                inputs, labels = data
                outputs = net(inputs)
                # zero the parameter gradients
                optimizer.zero_grad()
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()

            epoch_loss = running_loss / data_lengths[phase]
            curves[phase+"_loss"].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        t_a = train_accuracy()
        v_a = val_accuracy()
        curves["train_acc"].append(t_a)
        curves["val_acc"].append(v_a)
        print("train_acc: ", t_a)
        print("val_acc: ", v_a)

    return curves

curves = train_model(optimizer, criterion, 120)

print(curves)

plt.plot(list(range(120)),curves['train_acc'],list(range(120)),curves['val_acc'])
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title("Accuracy learning curve")
plt.legend("train", "accuracy")

# EPOCHS = 20
# train_losses = []
# train_acc = []
# val_acc = []
# for epoch in range(EPOCHS):  # loop over the dataset multiple times
#     running_loss = 0.0
#     epoch_loss = 0.0
#     net.train()
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         #statistics
#         running_loss += loss.item()
#         epoch_loss += loss.item()
#         if i % 200 == 199:  # print every 200 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0
#
#     net.eval()
#     train_acc.append(train_accuracy())
#     val_acc.append(val_accuracy())
#     epoch_loss = epoch_loss / len(trainloader)
#     train_losses.append(epoch_loss)
#     print("["+str(epoch)+"] train_loss: "+str(epoch_loss)+" - train_acc: "+str(train_acc[-1])+ " - val_acc: "+str(val_acc[-1]))
#
# print('Finished Training')

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))