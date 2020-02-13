# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>

from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import metrx as mx
import transformsx



# set the data directory for training models
DATA_DIR = 'data/tel'
BATCH_SIZE = 4
WORKERS = 6
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# check for cuda and set gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# data processing, adding additional sets and transforms, i.e. test corresponds to a directory
# and will create appropriate datasets
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transformsx.GausNoise(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])
}

# use the ImageFolder class from torchvision that accepts datasets structured by datax.py
# create datasets with appropriate transformation
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in data_transforms}
# create dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS) for x in data_transforms}

dataset_sizes = {x: len(image_datasets[x]) for x in data_transforms}
class_names = image_datasets['test'].classes
number_classes = len(class_names)


# utility method for displaying batches images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(IMAGE_NET_MEAN)
    std = np.array(IMAGE_NET_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
   
    # specify model to test
    MODEL_NAME = 'models/inception3TEL.pt'
    model = torch.load(MODEL_NAME)
    model.eval()
    model.to(device)
  
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    # confusion matrix
    confusion_matrix = torch.zeros(number_classes, number_classes)
    # loop through data
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # keep track of gradients for training only
        with torch.set_grad_enabled(False):
            # forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
           
            
            # compute loss
            loss = criterion(outputs, labels)
            
            # record 
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            
            # confusion matrix calculation
            for t, p in zip(predictions.view(-1), labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


    print("Confusion Matrix: \n", confusion_matrix)
    print("----------------------------------------")
    print("MX accuracy: ", mx.accuracy(confusion_matrix))
    print("MX target sensitivity: ", mx.sensitivity(confusion_matrix,0))
    print("MX class accuracy: ", mx.per_class_accuracy(confusion_matrix))
    print("class names: ", class_names)
    print("----------------------------------------")

