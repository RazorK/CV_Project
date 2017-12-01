import torch
import MySQLdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data.sampler as smp
#from tqdm import tqdm
from tqdm import tqdm as tqdm
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

imgTransform = transforms.Compose([transforms.Scale(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])
trainLoader, valLoader = dataLoader.get_train_valid_loader(cf.photo_url,50,32,'all',imgTransform,0.1,-1)

#define learningRate
learningRate = 1e-3 

#Definition of our loss.
#The MSELoss function 
criterion = nn.MSELoss()

new_pretrained_model = modified_resnet.resnet50(pretrained = True)
new_pretrained_model.fc = nn.Linear(512*4, 1)  

# Definition of optimization strategy.
optimizer = optim.SGD(new_pretrained_model.parameters(), lr = learningRate)

# Train the previously defined model.
result = train.train_model(new_pretrained_model, criterion, optimizer, trainLoader,
                           valLoader, n_epochs = 10, use_gpu = True, batch_size = 50, notebook = False, save_name = "resnet_all_category")
print result

torch.save(new_pretrained_model.state_dict(), "./resnet_all_category_final")