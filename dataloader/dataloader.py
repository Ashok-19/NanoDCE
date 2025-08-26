import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)

# Supported image formats
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']

def populate_train_list(lowlight_images_path):
    image_list_lowlight = []
    
    # Get all images with supported formats
    for extension in IMAGE_EXTENSIONS:
        image_list_lowlight.extend(glob.glob(os.path.join(lowlight_images_path, extension)))
        # Also check for uppercase extensions
        image_list_lowlight.extend(glob.glob(os.path.join(lowlight_images_path, extension.upper())))
    
    train_list = image_list_lowlight
    random.shuffle(train_list)
    
    return train_list

class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 512
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        
        try:
            data_lowlight = Image.open(data_lowlight_path)
            
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if data_lowlight.mode != 'RGB':
                data_lowlight = data_lowlight.convert('RGB')
            
            data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
            data_lowlight = (np.asarray(data_lowlight) / 255.0) 
            data_lowlight = torch.from_numpy(data_lowlight).float()
            
            return data_lowlight.permute(2, 0, 1)
            
        except Exception as e:
            print(f"Error loading image {data_lowlight_path}: {e}")
            # Return a blank image or skip this sample
            # You might want to implement better error handling here
            blank_image = torch.zeros(3, self.size, self.size)
            return blank_image

    def __len__(self):
        return len(self.data_list)