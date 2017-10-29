import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import MySQLdb
import configure as cf
from PIL import Image


class YelpDataSet(torch.utils.data.Dataset):
    def __init__(self, photo_dir, category, transform=None):
        self.photo_dir = photo_dir + '/photos'
        self.category = category
        self.transform = transform

        conn = MySQLdb.connect(host=cf.mysql_ip, user=cf.mysql_uesr, passwd=cf.mysql_pwd,
                               db=cf.mysql_db_name, charset="utf8")
        cursor = conn.cursor()

        # Get all photo id with the label = category
        self.photo_id = []
        print category
        cursor.execute('select id,business_id from photo where label=\'' + category + '\'')
        for row in cursor.fetchall():
            tem_dic = dict()
            tem_dic['id'] = row[0]
            tem_dic['business_id'] = row[1]
            self.photo_id.append(tem_dic)

        print('After search photo, find result: ' + str(len(self.photo_id)))
        print('Start search stars for each photo.')
        # Get the stars with exact business
        for dic in tqdm(self.photo_id):
            cursor.execute('select stars from business where id=\'' + dic['business_id'] + '\' LIMIT 1')
            row = cursor.fetchone()
            dic['stars'] = row[0]

            # self.photo_id: list of dict
            #    dict keys: id, business_id, stars

    def __len__(self):
        return len(self.photo_id)

    def __getitem__(self, idx):
        img_address = self.photo_dir + '/' + self.photo_id[idx]['id'] + '.jpg'
        image = Image.open(img_address).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # dic = {'image': image, 'stars': float(self.photo_id[idx]['stars'])}
        star = float(self.photo_id[idx]['stars'])
        return image, star
