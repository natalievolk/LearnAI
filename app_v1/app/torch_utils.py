import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(3,16,9)
        self.conv2 = nn.Conv2d(16, 64,11)
        self.conv3 = nn.Conv2d(64,256,13)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*18*18, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)
        self.softmax = nn.Softmax()

    # Defining the forward pass    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PATH = "app/classifier_restart.pt"
net = Net()
net.load_state_dict(torch.load(PATH, map_location='cpu'))
net.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize([225,225]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    # inputs = image_tensor.reshape(-1, 225*225)
    outputs = net(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
