
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from models import tiramisu
from datasets import shirts
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils
import os


CAMVID_PATH = os.path.join("gdrive","My Drive","image_extraction","data","tiramisu")
RESULTS_PATH = Path('gdrive/My Drive/tiramisu/results/')
WEIGHTS_PATH = Path('gdrive/My Drive/tiramisu/weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 3

mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
std = [0.27413549931506, 0.28506257482912, 0.28284674400252]


normalize = transforms.Normalize(mean=mean, std=std)
train_joint_transformer = transforms.Compose([
    joint_transforms.JointCenterCrop((512,224)), 
    joint_transforms.JointRandomHorizontalFlip()
    ])
test_joint_transformer = transforms.Compose([
    joint_transforms.JointCenterCrop((512,224))
    ])

train_dset = shirts.Shirts(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True)

val_dset = shirts.Shirts(
    CAMVID_PATH, 'val', joint_transform=test_joint_transformer,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=batch_size, shuffle=False)

test_dset = shirts.Shirts(
    CAMVID_PATH, 'test', joint_transform=test_joint_transformer,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=1, shuffle=False)

print("Train: %d" %len(train_loader.dataset.imgs))
print("Val: %d" %len(val_loader.dataset.imgs))
print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

utils.imgs.view_image(inputs[0])
utils.imgs.view_annotated(targets[0])

LR = 1e-4
LR_DECAY = 0.005*LR
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 20
torch.cuda.manual_seed(0)

#model = tiramisu.FCDenseNet67(n_classes=12).cuda()
model = tiramisu.FCDenseNet00(n_classes=2).cuda()
model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
#was criterion = nn.NLLLoss2d(weight=shirts.class_weight.cuda()).cuda()
criterion = nn.NLLLoss2d().cuda()

for epoch in range(1, N_EPOCHS+1):
    since = time.time()

    ### Train ###
    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###    
    train_utils.save_weights(model, epoch, trn_loss, trn_err)
    
    ### Test ###
    val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)    
    print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
    time_elapsed = time.time() - since  
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                     epoch, DECAY_EVERY_N_EPOCHS)
    
train_utils.test(model, test_loader, criterion, epoch=1)
train_utils.view_sample_predictions(model, test_loader, n=1)

