import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import dataLoader
import configure as cf
import train_function as train
import plot_utils as utils

# define transform function, define trainset and valset
# VGG-16 requires the input size of 224*224*3

imgTransform = transforms.Compose([transforms.Scale(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()])

trainLoader, valLoader = \
    dataLoader.get_train_valid_loader(cf.photo_url, 50, 32, 'food', imgTransform, 0.1, -1)

# define learningRate
learningRate = 5 * 1e-4

# Definition of our network.
network = models.vgg16(pretrained=True)
network.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 1),
)

# Definition of our loss.
# The MSELoss function
criterion = nn.MSELoss()

# Definition of optimization strategy.
optimizer = optim.SGD(network.parameters(), lr=learningRate)

result = []
# Train the previously defined model.
result = train.train_model(network, criterion, optimizer, trainLoader, valLoader,
                           n_epochs=10, use_gpu=True, notebook=False)
print result

utils.save_loss(result[2], result[3], './test_loss.png')
utils.save_accuracy(result[0],result[1], './test_accu.png')
