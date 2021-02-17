import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import requests
from io import BytesIO
from tqdm.notebook import tqdm

PATH = "classifier.pt"

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

'''
# TO FORMAT IMAGE
response = requests.get(self.df['jpg_url'][idx])
img = Image.open(BytesIO(response.content)).resize((225, 225))
pix = np.array(img)
try:
    pix = self.transform(pix)
except:
    pix = np.stack((np.array(pix, copy=True), np.array(pix, copy=True), np.array(pix, copy=True)), axis=2)
    pix = self.transform(pix)

return pix, int(self.df['classification'][idx])
'''

def get_category(image):