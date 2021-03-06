import numpy as np
import matplotlib.pyplot as plt

Void = [0,0,0]
Print = [255,255,255]
"""
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
"""
DSET_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
DSET_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]

label_colours = np.array([Void,Print])


def view_annotated(tensor, plot=True):
    temp = tensor.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,2):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = r#[:,:,0]
    rgb[:,:,1] = g#[:,:,1]
    rgb[:,:,2] = b#[:,:,2]
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
      
def get_annotated(tensor):
    temp = tensor.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,2):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = r#[:,:,0]
    rgb[:,:,1] = g#[:,:,1]
    rgb[:,:,2] = b#[:,:,2]
    
    return rgb

def decode_image(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array(DSET_MEAN)
    std = np.array(DSET_STD)
    inp = std * inp + mean
    return inp

def view_image(tensor):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

def get_image(tensor):
      inp = decode_image(tensor)
      #inp = np.clip(inp,0,1)
      return inp
