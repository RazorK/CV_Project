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

def class2id(name):
    if(name == 'food'):
        return 0
    elif (name == 'inside'):
        return 1
    elif (name == 'outside'):
        return 2
    elif (name == 'drink'):
        return 3
    elif (name == 'menu'):
        return 4
    else:
        return 5

class YelpDataSet(torch.utils.data.Dataset):
    def __init__(self, photo_dir, category, transform=None):
        self.photo_dir = photo_dir + '/photos'
        self.category = category
        self.transform = transform

        conn = MySQLdb.connect(host=cf.mysql_ip, user=cf.mysql_user, passwd=cf.mysql_pwd,
                               db=cf.mysql_db_name, charset="utf8")
        cursor = conn.cursor()

        # Get all photo id with the label = category
        self.photo_id = []
        print category
        if(category == 'all'):
            cursor.execute('select id, label from photo')
        else:
            cursor.execute('select id, label from photo where label=\'' + category + '\'')
        for row in cursor.fetchall():
            tem_dic = dict()
            tem_dic['id'] = row[0]
            tem_dic['label'] = row[1]
            self.photo_id.append(tem_dic)

            # self.photo_id: list of dict
            #    dict keys: id

    def __len__(self):
        return len(self.photo_id)

    def __getitem__(self, idx):
        img_address = self.photo_dir + '/' + self.photo_id[idx]['id'] + '.jpg'
        image = Image.open(img_address).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # dic = {'image': image, 'stars': float(self.photo_id[idx]['stars'])}
        label = self.photo_id[idx]['label']
        return image, class2id(label)

def get_train_valid_loader(photo_dir,
                               category,
                               batch_size=1,
                               random_seed=32,
                               transform=None,
                               valid_size=0.1,
                               set_num = -1,
                               shuffle=True,
                               num_workers=4,
                               pin_memory=False):

        #error_msg = "[!] valid_size should be in the range [0, 1]."
        #assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

        # load the dataset
        yelpDataset = YelpDataSet(photo_dir, category, transform)
        num_train = len(yelpDataset)
        indices = list(range(num_train))
        if set_num == -1:
            set_sum = num_train
        else:
            set_sum = set_num
        split = int(np.floor(valid_size * set_sum))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:set_sum], indices[:split]

        train_sampler = smp.SubsetRandomSampler(train_idx)
        valid_sampler = smp.SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(yelpDataset,
                                                   batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=num_workers, pin_memory=pin_memory)

        valid_loader = torch.utils.data.DataLoader(yelpDataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, valid_loader
#define learningRate
learningRate = 1e-3 

# Definition of our network.
network = modified_resnet.resnet50(pretrained = True)
network.fc = nn.Linear(512*4, 5)

#Definition of our loss.
#The MSELoss function 
criterion = nn.CrossEntropyLoss()

# Definition of optimization strategy.
optimizer = optim.SGD(network.parameters(), lr = learningRate)

imgTransform = transforms.Compose([transforms.Scale(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])

trainLoader, valLoader = get_train_valid_loader(cf.photo_url,'all', 50, 32,imgTransform,0.1,-1)

def train_model(network, criterion, optimizer, trainLoader, valLoader, batch_size = 50, n_epochs = 10, use_gpu = False):
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
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            if inputs.size(0) < batch_size or labels.size(0) < batch_size: continue
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
        train_accuracy.append(100 * correct / counter)
        train_loss.append(cum_loss / (1 + i))
        torch.save(network.state_dict(), 'res_clas_do' + str(epoch))
        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            if inputs.size(0) < batch_size or labels.size(0) < batch_size: continue
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
        val_accuracy.append(100 * correct / counter)
        val_loss.append(cum_loss / (1 + i))
    return [train_accuracy, val_accuracy, train_loss, val_loss]

result = train_model(network, criterion, optimizer, trainLoader,
                           valLoader, n_epochs = 10, use_gpu = True)

print result