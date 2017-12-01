from torch.autograd import Variable
import torch
from tqdm import tqdm_notebook as tqdm_nb
from tqdm import tqdm


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
            
        train_accuracy.append(temp_accuracy)
        train_loss.append(temp_loss)
        
        if(save_name != 'default'):
            torch.save(network.state_dict(), save_name + str(epoch))
        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc='Validation epoch %d' % epoch)
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
            outputs = network(inputs)
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

        val_accuracy.append(temp_accuracy)
        val_loss.append(temp_loss)
    return [train_accuracy, val_accuracy, train_loss, val_loss]
