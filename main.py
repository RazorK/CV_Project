import torch
from dataset import TryDataSet
import configure as cf
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = TryDataSet(cf.photo_url, 'food')
    sample = a[0]

    print sample['stars']

    pil2tensor = transforms.ToTensor()
    img = pil2tensor(sample['image'])

    def plot_image(tensor):
        plt.figure()
        plt.imshow(tensor.numpy().transpose(1, 2, 0))
        plt.show()

    plot_image(img)

