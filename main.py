import torch
from dataset import YelpDataSet
import configure as cf
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import dataLoader as load

if __name__ == "__main__":
    train, val = load.get_train_valid_loader(cf.photo_url, 50, 32, 'food')
    print train.__len__()
    print val.__len__()
