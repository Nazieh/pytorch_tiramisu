import os
import sys
import math
import string
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

import cv2 as cv

from . import imgs as img_utils

#RESULTS_PATH = '.results/'
#WEIGHTS_PATH = '.weights/'

RESULTS_PATH = 'gdrive/My Drive/tiramisu/results/'
WEIGHTS_PATH = 'gdrive/My Drive/tiramisu/weights/'

def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return np.round(err,5)

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        trn_loss += loss.data.item()
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())
        if (idx+1)%100 == 0:
            print("{} batches done, loss: {}, error: {}.".format(idx+1,trn_loss,trn_error))
        
    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])

class Case:
    def __init__(self,inp,target,pred,idx):
        self.inp = inp
        self.target = target
        self.pred = pred
        self.idx = idx
        
        
def test_set_predictions(model, loader):
    cases = []
    idx = 0
    for inputs, targets in iter(loader):
        data = Variable(inputs.cuda(), volatile=True)
        label = Variable(targets.cuda())
        output = model(data)
        pred = get_predictions(output)
        batch_size = inputs.size(0)
        case = Case(inputs[0],targets[0],pred[0],idx)
        idx+=1
        cases.append(case)
    
    for case in cases:
        in_im =  img_utils.get_image(case.inp)
        target_im = img_utils.get_annotated(case.target)
        pred_im =  img_utils.get_annotated(case.pred)
        cv.imwrite(os.path.join(RESULTS_PATH,f"{case.idx}.png"),in_im)
        cv.imwrite(os.path.join(RESULTS_PATH,f"{case.idx}_annot.png"),target_im)
        cv.imwrite(os.path.join(RESULTS_PATH,f"{case.idx}_pred.png"),pred_im)
       
