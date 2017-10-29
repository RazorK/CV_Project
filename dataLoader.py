import torch
import torchvision.transforms as transforms
import torch.utils.data.sampler as smp
import dataset
import numpy as np


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           category,
                           transform=None,
                           valid_size=0.1,
                           set_num = -1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Source:
    https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - category: the category for the yelp dataset picture
    - transform: transform function for pictures
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load the dataset
    yelpDataset = dataset.YelpDataSet(data_dir, category, transform)
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
