from torch.utils.data import Dataset
from create_fontstyle import fontstyle_list
import torch

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import glob
import random
import os

font_folder = 'font'
font_name = ['arial', 'bodoni','calibri','futura','heveltica','times-new-roman']

fonts = fontstyle_list(font_folder, font_name)

class digitalMNIST(Dataset):
    """
    Generate digital mnist dataset for digits recognition
    """
    def __init__(self, samples, random_state, transform = None):
        self.samples = samples
        self.random_state = random_state
        self.transfrom = transform
        self.fonts = fonts

        random.seed(random_state)
    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        color = random.randint(200,255)
        #Generate image
        img = Image.new("L",(256, 256))
        label = random.randint(0,9)
        size = random.randint(180, 220)
        x = random.randint(60, 80)
        y = random.randint(30, 60)

        draw = ImageDraw.Draw(img)
        #Choose random font style in font style list
        font = ImageFont.truetype(random.choice(self.fonts), size)
        draw.text((x,y), str(label), color, font = fonts)
        
        img = img.resize((28,28), Image.BILINEAR)
        if self.transfrom:
            img = self.transfrom(img)
        return img, label

class AddSPNoise(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, tensor):
        sp = (torch.rand(tensor.size()) < self.prob) * tensor.max()
        return tensor + sp

    def __repr__(self):
        return self.__class__.__name__ + "(prob={0})".format(self.prob)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )