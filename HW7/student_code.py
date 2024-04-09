# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        # 3 input image channel (color), 6 output channels, 5x5 square convolution
        self.Conv2d1 = nn.Conv2d(3, 6, 5)
        self.Conv2d2 = nn.Conv2d(6, 16, 5)
        #y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 256) #32x32 dimension
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, num_classes) #Output is num_classes
        #Max_Pool layer (kernal size = 2, stride = 2)
        self.maxpool = nn.MaxPool2d(2,2)
        #Flatten
        self.flat = nn.Flatten()
        #Relu
        self.relu = nn.ReLU()

    def forward(self, x):
        shape_dict = {}
        # certain operations
        # Stage 1
        x = self.Conv2d1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        shape_dict[1] = list(x.size()) # Add stage 1 into dict

        # Stage 2
        x = self.Conv2d2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        shape_dict[2] = list(x.size()) # Add stage 2 into dict

        x = self.flat(x) # Flatten it (stage 3)
        shape_dict[3] = list(x.size()) # Add stage 3 into dict

        # Stage 4
        x = self.fc1(x)
        x = self.relu(x)
        shape_dict[4] = list(x.size()) # Add stage 4 into dict

        # Stage 5
        x = self.fc2(x)
        x = self.relu(x)
        shape_dict[5] = list(x.size()) # Add stage 5 into dict

        x = self.fc3(x) # Stage 6
        shape_dict[6] = list(x.size()) # Add stage 6 into dict
        
        return x, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    model_params = sum(i.numel() for i in model.parameters() if i.requires_grad)
    model_params /= 1e6

    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
