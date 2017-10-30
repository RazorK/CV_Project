import torch
import MySQLdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data.sampler as smp
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import IPython.display
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import dataset
import dataLoader
import configure as cf


#define transform function, define trainset and valset
#ResNet50 requires the input size of 256*256*3

imgTransform = transforms.Compose([transforms.Scale(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])

trainLoader, valLoader = dataLoader.get_train_valid_loader(cf.photo_url,50,32,'food',imgTransform,0.1,-1)


%matplotlib inline
import matplotlib.pyplot as plt
def plot_loss(train_loss,val_loss):
    plt.plot(train_loss,'r',label = 'train loss')
    plt.plot(val_loss,'b',label = 'validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_accuracy(train, val):
    plt.plot(train,'r',label = 'train accuracy')
    plt.plot(val,'b',label = 'validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
#define train model
def train_model(network, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = True):
    
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        correct = 0.0
        cum_loss = 0.0
        counter = 0

        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (inputs, stars)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            stars = Variable(stars.type(torch.FloatTensor))
            if inputs.size(0)<50 or stars.size(0)<50: break
            
            if use_gpu:
                inputs = inputs.cuda()
                stars = stars.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, stars)

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            #set a rule: if prediction values is between real_value-0.5 and real_value+0.5, correct+1
            cum_loss += loss.data[0]
            pre_star = outputs.data
            larger = (pre_star.view(50) >= (stars.data-0.5)).type(torch.IntTensor)
            littler = (pre_star.view(50) <= (stars.data+0.5)).type(torch.IntTensor)
            correct += (larger+littler).eq(2).sum() 
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
        
        train_accuracy.append(100 * correct / counter)
        train_loss.append(cum_loss / counter)

        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, stars)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            stars = Variable(stars.type(torch.FloatTensor))
            if inputs.size(0)<50 or stars.size(0)<50: break
            
            if use_gpu:
                inputs = inputs.cuda()
                stars = stars.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, stars)

            # logging information.
            cum_loss += loss.data[0]
            pre_star = outputs.data
            larger = (pre_star.view(50) >= (stars.data-0.5)).type(torch.IntTensor)
            littler = (pre_star.view(50) <= (stars.data+0.5)).type(torch.IntTensor)
            correct += (larger+littler).eq(2).sum() 
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
        
        val_accuracy.append(100 * correct / counter)
        val_loss.append(cum_loss / counter)
    return [train_accuracy,val_accuracy,train_loss,val_loss]
            

#define learningRate
learningRate = 1e-3 

# Definition of our network.
network = models.resnet50(pretrained = True)
network.fc = nn.Linear(512*4, 1)  

#Definition of our loss.
#The MSELoss function 
criterion = nn.MSELoss()

# Definition of optimization strategy.
optimizer = optim.SGD(network.parameters(), lr = learningRate)

result = []
# Train the previously defined model.
result = train_model(network, criterion, optimizer, trainLoader, valLoader, n_epochs = 20, use_gpu = True)

plot_loss(result[2],result[3])
plot_accuracy(result[0],result[1])
