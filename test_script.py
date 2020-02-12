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

test_joint_transformer = transforms.Compose([
    joint_transforms.JointCenterCrop((512,224))
    ])

test_dset = shirts.Shirts(
    CAMVID_PATH, 'test', joint_transform=test_joint_transformer,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=1, shuffle=False)

print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(test_loader.dataset.classes))



torch.cuda.manual_seed(0)

model = torch.load(os.path.join(WEIGHTS_PATH,"weights-20-0.217-0.000.pth"))
model.eval()
train_utils.test_set_predictions(model, test_loader)
