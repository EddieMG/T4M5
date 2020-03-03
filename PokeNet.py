import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter



class PokeNet(nn.Module):
    def __init__(self):
        super(PokeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5,stride=4)
        self.BN1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(16, 16, 3)
        self.BN2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(16, 64, 3)
        self.BN3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.Drop = nn.Dropout2d() #0.5 by default

        self.conv4 = nn.Conv2d(64, 8, 1)
        self.BN4 = nn.BatchNorm2d(8)
        
        self.pool4 = nn.AvgPool2d(5)
        self.fc = nn.Linear(8,8)

    def forward(self, x):
       # print(x.size())
        x = self.pool1(F.relu(self.BN1(self.conv1(x))))
        x = self.pool2(F.relu(self.BN2(self.conv2(x))))
        x = self.Drop(self.pool3(F.relu(self.BN3(self.conv3(x)))))
        x = F.relu(self.BN4(self.conv4(x)))
        #print(x.size())
        x = self.pool4(x)
        #print("before softmax")
        #print(x.size())
        x = x.view(-1,8*1*1)
        x = self.fc(x)
        #x = F.softmax(x,dim=1)
        #print("after softmax")
        #print(x.size())
        return x


class Switcher(object):
    def trainloader(self, argument):
        """Dispatch method"""
        method_name = 'trainloaderNumber' + str(argument)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid argument")
        # Call the method as we return it
        return method()
    
    def testloader(self, argument):
        """Dispatch method"""
        method_name = 'trainloaderNumber' + str(argument)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid argument")
        # Call the method as we return it
        return method()
 
    def trainloaderNumber0(self):
        print('using 0')
        return trainloader0   
    def trainloaderNumber1(self):
        print('using 1')
        return trainloader1
    def trainloaderNumber2(self):
        print('using 2')
        return trainloader2
    def trainloaderNumber3(self):
        print('using 3')
        return trainloader3
    def trainloaderNumber4(self):
        print('using 4')
        return trainloader4       

    def testloaderNumber0(self):
        print('using 0')
        return testloader0   
    def testloaderNumbe1(self):
        print('using 1')
        return testloader1
    def testloaderNumbe2(self):
        print('using 2')
        return testloader2
    def testloaderNumbe3(self):
        print('using 3')
        return testloader3
    def testloaderNumbe4(self):
        print('using 4')
        return testloader4                 
"""

Train the model

"""

def main():

    for epoch in range(4):  # loop over the dataset multiple times
        net.train()
        a = Switcher()
        trainloader = a.trainloader(epoch % 5)
        testloader = a.testloader(epoch % 5)
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data
            #print("lables")
            #print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        #writer.add_scalar('Loss/train', running_loss, epoch)
        with torch.no_grad():
            net.eval()
            for data in trainloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # writer.add_scalar('Accuracy/train', (100 * correct / total), epoch)
            print( 'Accuracy of the network on the epoch: %d %%'%(100 * correct / total))
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            net.eval()
            for data in testloader:
                images, labels = data
                outputs = net(images)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            #writer.add_scalar('Val_Loss/train', val_loss, epoch)
            print("Val loss "+ str(val_loss))
        # writer.add_scalar('val_Accuracy/train', (100 * correct / total), epoch)
            print( 'Accuracy of the network on the validation data: %d %%'%(100 * correct / total))
        
    print('Finished Training')
    testset = torchvision.datasets.ImageFolder(root=testset_path, transform = transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for _,data in enumerate(testloader, 0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 807 test images: %d %%' % (
        100 * correct / total))
        

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        net.eval()
        for _,data in enumerate(testloader, 0):
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

if __name__ == "__main__":
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    #writer.add_graph(net, images)
    #writer.close()
    """
    Creating the dataset and the dataloader. 
    Remember to override the __getitem__ and __len__

    """

    trainset_path =r'C:\Users\Eduard\Documents\M3\Databases\MIT_split\train'
    testset_path =r'C:\Users\Eduard\Documents\M3\Databases\MIT_split\test'

    trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    testset = torchvision.datasets.ImageFolder(root=testset_path, transform = transforms.ToTensor())
    trainset = torchvision.datasets.ImageFolder(root=trainset_path , transform = trans)
    #print(trainset.imgs[0:10])
    random.shuffle(trainset.imgs)
    #print(trainset.imgs[0:10])

    folds = [trainset.imgs[x:x+376] for x in range(0, len(trainset.imgs), 376)]

    trainset.imgs= folds[0]+folds[1]+folds[2]+folds[3]
    testset.imgs = folds[4]
    trainloader0 = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=2)
    testloader0 = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)
    print(len(trainset.imgs))
    print(len(testset.imgs))
    trainset.imgs= folds[1]+folds[2]+folds[3]+folds[4]
    testset.imgs = folds[0]
    trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=2)
    testloader1 = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)

    trainset.imgs= folds[2]+folds[3]+folds[4]+folds[0]
    testset.imgs = folds[1]
    trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=2)
    testloader2 = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)                                                                        

    trainset.imgs= folds[3]+folds[4]+folds[0]+folds[1]
    testset.imgs = folds[2]
    trainloader3 = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=2)
    testloader3 = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)

    trainset.imgs= folds[4]+folds[0]+folds[1]+folds[2]
    testset.imgs = folds[3]
    trainloader4 = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=2)
    testloader4 = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)



    #writer = SummaryWriter()


    net = PokeNet()


    """

    Set up an optimizer and define a loss function

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters(),lr=0.1, rho=0.9, eps=1e-06, weight_decay=0)
    classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
    """

    Creating the PokeNet

    """
    main()