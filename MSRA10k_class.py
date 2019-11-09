from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import zipfile
import time
class MSRAData(Dataset):
    def __init__(self ,x ,y ,transform=None, xform_label=None):

        self.images = x           
        self.labels = y
      
        self.transform = transform
        self.label_transform = xform_label                 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
   
        sample = self.images[idx,:]
        labels = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.label_transform:
            labels = self.label_transform(labels)
        return sample, labels


def extract(file_name):
  start_time = time.time()
  with zipfile.ZipFile(file_name, 'r') as zip:
    zip.extractall()
    print("Extracted")
    duration = time.time() - start_time
    print(duration, "s")

def extract_data(x_pth, y_pth):
	X = np.load(x_pth)
	Y = np.load(y_pth)
	print('Done!')
	return X, Y


def get_trainloader(X, Y):
	transformations = transforms.Compose([
          
          transforms.ToPILImage(),
          transforms.Resize((256,256)),
          transforms.ToTensor(),          
          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
          
          ])
	labels_xform = transforms.Compose([
	          transforms.ToPILImage(),
	          transforms.Grayscale(num_output_channels=1),
	          transforms.ToTensor()
	])

	train_set = MSRAData(X[:8000], Y[:8000], transformations, labels_xform) 
	valid_set = MSRAData(X[-2000:], Y[-2000:], transformations, labels_xform) 

	train_loader = torch.utils.data.DataLoader(
	    dataset=train_set,
	    batch_size=8,
	    shuffle=True
	)

	valid_loader = torch.utils.data.DataLoader(
	    dataset=valid_set,
	    batch_size=1,
	    shuffle=False
	)
	return train_loader, valid_loader