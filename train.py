# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# Misc Imports
import numpy as np
import time
import sys
import os
# Google Drive Import
from google.colab import drive

# Training The Model using a function
def train_model(model, train_loader, criterion, optimizer, num_epochs, save_pth, load_point, device):
    since = time.time()
    losses = { x: [] for x in ['train']}
    
    
    full_pth = save_pth + str(load_point) +'.pth'
    if os.path.exists(full_pth):      
      checkpoint = torch.load(full_pth)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      losses = checkpoint['loss']
      epoch = checkpoint['epoch']
      print('model loaded')
    else:
      epoch = 0
        
    
    for epoch in range(epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 
        
        if not os.path.exists(save_pth):
          drive.mount('/content/drive/')
        
        tgt_pth = save_pth + str(epoch) +'.pth'
        
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': losses
          }, tgt_pth)
          

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(train_loader):
                completion = (i/len(train_loader))*100
                print('Epoch: {} - {:.2f}% Complete'.format(epoch, completion))
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.long().squeeze()
  
               
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # enable only if training - saves memory
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs, list_predictions = model(inputs)
                    
                    
                    loss = criterion(outputs.float(), labels)
                    
                    for prediction in list_predictions:
                      loss+= criterion(prediction.float(), labels)
                    
                    if completion % 5 ==0:
                      print('loss: {:.2f}'.format(loss.data.item()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            

                
            epoch_loss = running_loss / 8000


            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))
            
            # For graphing purposes
            losses[phase].append(epoch_loss)            
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
 
    return model, losses