import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


# We first get the data loaders of either train_set or test_set depending on the input (true/false) and output the loader
def get_data_loader(training = True):
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform) #Train set
    test_set = datasets.FashionMNIST('./data',train=False, download=False,transform=custom_transform) #Test set

    #Check if we are getting train or test
    if(training == True):
        return torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)


# This function builds an untrained neural network model
def build_model(): 
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10))


# This function implements the training prodecure
def train_model(model, train_loader, criterion, T):

    opt = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) # An optimizer
    model = model.train() # Sets the model to train mode

    # Loop through the epochs
    for i in range(0, T):
        running_loss = 0.0 # Stores the running loss for each epoch
        correct = 0 # Stores how many are correct for each epoch
        total = 0 # Stores the total for each epoch
        for j, data in enumerate(train_loader, 0):
            inputs,labels = data # Get inputs from data [inputs, labels]

            opt.zero_grad() # Zero the parameter gradiants
            
            # Foward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()*labels.size(0) # Add the loss * batch size to the accumulating loss

            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print(f'Train Epoch: {i} Accuracy: {correct}/{total}({(100*correct/total):.2f}%) Loss: {running_loss/total:.3f}')



def evaluate_model(model, test_loader, criterion, show_loss = True):
    model = model.eval() # Turn model into evaluation mode

    correct = 0 # Stores the total number of correct
    total = 0 # Stores the total
    running_loss = 0.0 # Stores the accumulating loss

    # Doesn't track gradiants
    with torch.no_grad():
        # For every data in test_loader
        for data in test_loader:
            images, labels = data # Get inputs from data [images, labels]

            outputs = model(images) # Calculate output by running images through the network

            loss = criterion(outputs, labels)

            running_loss += loss.item()*labels.size(0) # Add the loss * batch size to the accumulating loss

            # The class with highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Check if we show the average loss
    if show_loss == True:
        print(f'Average Loss: {running_loss/total:.4f}')

    print(f'Accuracy: {100*correct/total:.2f}%')
    

# This function predicts the labels
def predict_label(model, test_images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot'] # All the class names

    img = test_images[index] # Get the specific image
    input_tensor = img.clone().detach() # Copies the tensor

    output_tensor = model.forward(input_tensor) # Gets the output from input tensor

    prob = F.softmax(output_tensor, dim = 1) # Get the probabilities 

    values, indexes = torch.topk(prob, 3) # Get the top 3 values and indexes

    # Print the class_name correlating with the index with the value as a % in decreasing order
    for i in range(3):
        print(f'{class_names[indexes[0][i].item()]}: {values[0][i].item()*100:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

    # Test first function (Get the DataLoader)
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)

    test_loader = get_data_loader(False)
    print(type(test_loader))
    print(test_loader.dataset)

    #Test second function (Build your Model)
    model = build_model()
    print(model)

    #Test third function (Train your Model)
    criterion = nn.CrossEntropyLoss()
    T = 5
    train_model(model, train_loader, criterion, T)

    #Test fouth function (Evaluate your model)
    evaluate_model(model, train_loader, criterion)

    #Test fifth function (Predict_label) WIP
    test_images, test_labels = next(iter(test_loader))
    predict_label(model, test_images, 1) 
