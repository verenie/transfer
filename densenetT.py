from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import metrx as mx



# set the data directory for training models
# this is where we select which fold to train on
DATA_DIR = 'data/tel'
LOG_DIR = 'logs'
# progress file name will be written to the logs dir
PROGRESS_FILE = 'DENSENETex.csv'
BATCH_SIZE = 6
WORKERS = 6
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
# freeze layers or finetune the whole model
FREEZE = True
# model selection based on accuracy or sensitivity
SELECT_ON_ACC = True

# check for cuda and set gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# data processing, adding additional sets and transforms, i.e. test corresponds to a directory
# and will create appropriate datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)

    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
class_names = image_datasets['train'].classes
number_classes = len(class_names)

def model_selection(model, criterion, optimizer, scheduler, epochs=30):
    # time execution
    start = time.time()

    # store best model weights
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_sen = 0.0
    gt_file = os.path.join(LOG_DIR,PROGRESS_FILE)
    for epoch in range(epochs):
        print('Epoch: ',epoch)
        for stage in ['train', 'val']:
            if stage == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            
            
            confusion_matrix = torch.zeros(number_classes, number_classes)
            # loop through data
            for inputs, labels in dataloaders[stage]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero out gradients
                optimizer.zero_grad()
                # keep track of gradients for training only
                with torch.set_grad_enabled(stage == 'train'):
                    # forward pass
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    # compute loss
                    loss = criterion(outputs, labels)
                    if stage == 'train':
                        loss.backward()
                        optimizer.step()
                # record
                # confusion matrix calculation
                for t, p in zip(predictions.view(-1), labels.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1 
                total_loss += loss.item() * inputs.size(0)
            if stage == 'train':
                scheduler.step()
            epoch_loss = total_loss / dataset_sizes[stage]
            epoch_acc =  mx.accuracy(confusion_matrix).item()
            # get sensitivity for our target class 0
            epoch_sen = mx.sensitivity(confusion_matrix,0).item()
            # print out loss, acc, confusion matrix
            print("Stage: ",stage," epoch loss: ", epoch_loss, " epoch acc: ", epoch_acc, " epoch sen: ", epoch_sen)
            print(confusion_matrix)
             # append to progress.csv for training
            if stage == 'val':
              with open(gt_file, 'a') as gt:
                writer = csv.writer(gt)
                if SELECT_ON_ACC:
                    writer.writerow([epoch, epoch_acc])
                else:
                    writer.writerow([epoch, epoch_sen])
            # when validating, check if current model is best
            if SELECT_ON_ACC:
              if stage == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
            else:
              if stage == 'val' and epoch_sen > best_sen:
                
                best_sen = epoch_sen
                best_weights = copy.deepcopy(model.state_dict())
    
    spent = time.time() - start
    print('Training and selection complete in {:.0f}m {:.0f}s'.format(spent // 60, spent % 60))
    if SELECT_ON_ACC:
        print('Best val accuracy: {:4f}'.format(best_acc))
    else:
        print('Best val sensitivity: {:4f}'.format(best_sen))
    # return the best model
    model.load_state_dict(best_weights)
    return model





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
    
    # select a ResNetX architecture
    model_conv = torchvision.models.densenet161(pretrained=True)
    # freeze the layers we want
    if FREEZE:
        for param in model_conv.parameters():
            param.requires_grad = False
    # set new layer (densenet final layer is classifier)
    num_features = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_features,2)
    model_conv.to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizing only the final layer
    if FREEZE:
        optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    # learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7,gamma=0.1)
    # start model selction
    model_conv = model_selection(model_conv, criterion, optimizer_conv, exp_lr_scheduler, epochs=30)
    # save the model
    torch.save(model_conv,'models/densenet161TELex.pt')