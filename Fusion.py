import torch
import MySQLdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data.sampler as smp
from tqdm import tqdm
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
import plot_utils as utils
import train_function as train
import resnet as modified_resnet

new_pretrained_model = modified_resnet.resnet50(pretrained = True)
new_pretrained_model.fc = nn.Linear(512*4, 1)  
new_pretrained_model.load_state_dict(torch.load('./Results/AllCategory/resnet_all_best'))
new_pretrained_model = new_pretrained_model.eval()

classifier = modified_resnet.resnet50(pretrained = True)
classifier.fc = nn.Linear(512*4, 5)
classifier.load_state_dict(torch.load('./Classification/Classification_Dropout/res_clas_do6'))
classifier = classifier.eval()

imgTransform = transforms.Compose([transforms.Scale(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])
trainLoader, valLoader = dataLoader.get_train_valid_loader(cf.photo_url,50,32,'all',imgTransform,0.1,500)

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.fusion = nn.Linear(4096, 1)
        
    def forward(self, x, classify, regress):
        # classifiy resnet feature extraction
        f1 = classify.conv1(x)
        f1 = classify.bn1(f1)
        f1 = classify.relu(f1)
        f1 = classify.maxpool(f1)

        f1 = classify.layer1(f1)
        f1 = classify.layer2(f1)
        f1 = classify.layer3(f1)
        f1 = classify.layer4(f1)

        f1 = classify.avgpool(f1)
        f1 = f1.view(x.size(0), -1)
        
        # regression resnet feature extraction
        f2 = regress.conv1(x)
        f2 = regress.bn1(f2)
        f2 = regress.relu(f2)
        f2 = regress.maxpool(f2)

        f2 = regress.layer1(f2)
        f2 = regress.layer2(f2)
        f2 = regress.layer3(f2)
        f2 = regress.layer4(f2)

        f2 = regress.avgpool(f2)
        f2 = f2.view(x.size(0), -1)
        
        feature = torch.cat((f1,f2),1)
        result = self.fusion(feature)
        return result
        
fusionModel = FusionModel()

fusionModel.load_state_dict(torch.load('./resnet_all_first'))
classifier = classifier.eval().cuda()
new_pretrained_model = new_pretrained_model.eval().cuda()
# define train model
def train_model(network, criterion, optimizer, trainLoader, valLoader,
                n_epochs=10, use_gpu=True, batch_size=50, notebook=True, save_name = 'default'):
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
        temp_accuracy = 0
        temp_loss = 0

        # Make a pass over the training data.
        if notebook:
            t = tqdm_nb(trainLoader, desc='Training epoch %d' % epoch)
        else:
            t = tqdm(trainLoader, desc='Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (inputs, stars)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            stars = Variable(stars.type(torch.FloatTensor))
            if inputs.size(0) < batch_size or stars.size(0) < batch_size: continue

            if use_gpu:
                inputs = inputs.cuda()
                stars = stars.cuda()

            # Forward pass:
            outputs = network(inputs, classifier, new_pretrained_model)
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
            # set a rule: if prediction values is between real_value-0.5 and real_value+0.5, correct+1
            cum_loss += loss.data[0]
            pre_star = outputs.data
            larger = (pre_star.view(batch_size) >= (stars.data - 0.5)).type(torch.IntTensor)
            littler = (pre_star.view(batch_size) <= (stars.data + 0.5)).type(torch.IntTensor)
            correct += (larger + littler).eq(2).sum()
            counter += inputs.size(0)
            temp_accuracy = 100 * correct / counter
            temp_loss = cum_loss / (1 + i)
            t.set_postfix(loss=temp_loss, accuracy=temp_accuracy)
            del inputs, stars
            
        train_accuracy.append(temp_accuracy)
        train_loss.append(temp_loss)
        
#         if(save_name != 'default'):
#             torch.save(network.state_dict(), save_name + str(epoch))
        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        if notebook:
            t = tqdm_nb(valLoader, desc='Val epoch %d' % epoch)
        else:
            t = tqdm(valLoader, desc='Val epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, stars)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            stars = Variable(stars.type(torch.FloatTensor))
            if inputs.size(0) < batch_size or stars.size(0) < batch_size: continue

            if use_gpu:
                inputs = inputs.cuda()
                stars = stars.cuda()

            # Forward pass:
            outputs = network(inputs, classifier, new_pretrained_model)
            loss = criterion(outputs, stars)

            # logging information.
            cum_loss += loss.data[0]
            pre_star = outputs.data
            larger = (pre_star.view(batch_size) >= (stars.data - 0.5)).type(torch.IntTensor)
            littler = (pre_star.view(batch_size) <= (stars.data + 0.5)).type(torch.IntTensor)
            correct += (larger + littler).eq(2).sum()
            counter += inputs.size(0)
            temp_accuracy = 100 * correct / counter
            temp_loss = cum_loss / (1 + i)
            t.set_postfix(loss=temp_loss, accuracy=temp_accuracy)
            del inputs, stars

        val_accuracy.append(temp_accuracy)
        val_loss.append(temp_loss)
    return [train_accuracy, val_accuracy, train_loss, val_loss]

#define learningRate
learningRate = 1e-3 

#Definition of our loss.
#The MSELoss function 
criterion = nn.MSELoss()

# Definition of optimization strategy.
optimizer = optim.SGD(fusionModel.parameters(), lr = learningRate)

# Train the previously defined model.
result = train_model(fusionModel, criterion, optimizer, trainLoader,
                           valLoader, n_epochs = 10, use_gpu = True, batch_size = 50, notebook = False, save_name = "resnet_all_category")
print result
