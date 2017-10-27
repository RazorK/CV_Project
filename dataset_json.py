# abandoned too slow..

import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import pprint
from torchvision import transforms, utils


class TryDataset(torch.utils.data.Dataset):
    def __init__(self, photo_dir, json_dir, category, transform=None):
        self.bus_dir = json_dir + '/business.json'
        self.photo_json_dir = photo_dir + '/photos.json'
        self.photo_dir = photo_dir + '/photos'
        self.category = category
        self.transform = transform

        # Get all photo id with the label = category
        self.photo_id = []
        f = open(self.photo_json_dir, 'r')
        for line in f.readlines():
            dic = json.loads(line)
            if dic['label'] == category:
                self.photo_id.append(dic)
        print('after searching, get:'+ str(len(self.photo_id)))

        # Find the star for the specific business that the photo belongs
        f = open(self.bus_dir, 'r')
        for dic in tqdm(self.photo_id):
            bid = dic['business_id']
            f.seek(0)
            for line in f.readlines():
                bdic = json.loads(line)
                if bdic['business_id'] == bid:
                    dic['stars'] = bdic['stars']
        for i in range(10):
            print self.photo_id[i]



    def __len__(self):
        pass
        # return len(self.sname)

    def __getitem__(self, idx):
        pass
        # img_name = self.pwdroot + '/' + self.sname[idx]
        # label = class2id(self.sname[idx].split('.')[0])
        # image = Image.open(img_name).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        # return (image, label)
