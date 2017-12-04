import torch
import MySQLdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data.sampler as smp
#from tqdm import tqdm
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
from PIL import Image

import dataset
import dataLoader
import configure as cf
import plot_utils as utils
import train_function as train
import resnet as modified_resnet

imgTransform = transforms.Compose([transforms.Scale(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])
trainLoader, valLoader = dataLoader.get_train_valid_loader(cf.photo_url,1,32,'all',imgTransform,0.1,-1)

# define classification network
classification = modified_resnet.resnet50(pretrained = True)
classification.fc = nn.Linear(512*4, 5)
classification.load_state_dict(torch.load('./res_clas_do7'))

#define regression_food
regress_food = modified_resnet.resnet50(pretrained = True)
regress_food.fc = nn.Linear(512*4, 1)
regress_food.load_state_dict(torch.load('./test_FOOD_ResNet_Final'))
#define regression_drink
regress_drink = models.resnet50(pretrained = False)
regress_drink.fc = nn.Linear(512*4, 1)
regress_drink.load_state_dict(torch.load('./test_DRINK_ResNet_Final'))
#define regression_inside
regress_inside = models.resnet50(pretrained = False)
regress_inside.fc = nn.Linear(512*4, 1)
regress_inside.load_state_dict(torch.load('./test_INSIDE_ResNet_Final'))
#define regression_outside
regress_outside = models.resnet50(pretrained = False)
regress_outside.fc = nn.Linear(512*4, 1)
regress_outside.load_state_dict(torch.load('./test_OUTSIDE_ResNet_Final'))
#define regression_menu
regress_menu = models.resnet50(pretrained = False)
regress_menu.fc = nn.Linear(512*4, 1)
regress_menu.load_state_dict(torch.load('./test_MENU_ResNet_Final'))

#define a demo for validation
#probably here is something wrong with classification netwrok - discuss it this afternoon
n_epochs = 5
for epoch in range(0, n_epochs):
    correct = 0
    counter = 0
    
    t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
    for (i, (inputs, stars)) in enumerate(t):
        inputs = Variable(inputs)
        stars = Variable(stars.type(torch.FloatTensor))
        
        C_outputs = classification(inputs)
        max_scores, max_labels = C_outputs.data.max(1)
  
        if (max_labels == torch.LongTensor([0])).numpy():
            result = regress_food(inputs)
        elif (max_labels == torch.LongTensor([1])).numpy():
            result = regress_inside(inputs)
        elif (max_labels == torch.LongTensor([2])).numpy():
            result = regress_outside(inputs)
        elif (max_labels == torch.LongTensor([3])).numpy():
            result = regress_drink(inputs)
        elif (max_labels == torch.LongTensor([4])).numpy():
            result = regress_menu(inputs)
        else:
            result = 0
        
        pre_star = result.data
        
        larger = (pre_star >= (stars.data-0.5)).type(torch.IntTensor)
        littler = (pre_star <= (stars.data+0.5)).type(torch.IntTensor)
        correct += (larger+littler).eq(2).sum()
        counter += 1
        
        t.set_postfix(accuracy = 100 * correct / counter)