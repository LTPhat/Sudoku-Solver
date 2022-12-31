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
        target = random.randint(0,9)
        size = random.randint(150, 250)
        x = random.randint(60, 90)
        y = random.randint(30, 60)
        color = random.randint(200,255)

        #Generate image
        
        return super().__getitem__(index)