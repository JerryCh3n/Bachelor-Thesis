# Importing libraries
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transform

import numpy as np

import matplotlib.pylab as plt

torch.manual_seed(42)

use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

#-----Class to represent dataset-----
class lineDataSet():

    def __init__(self):

        # Loading the csv file from the folder path
        # First row is data labels so delete
        data1 = np.loadtxt('cleanlinedata.csv', delimiter=',',
                           dtype=np.float32, skiprows=1)

        # First 4 column are class parameters
        # Last 2 are line properties
        # Data in Each Column
        # Q (uL/min), Vg (mm/min), LDR (μL/mm), Print Height (mm), Average Width (μm), Average Thickness (μm)

        # Zero center and normalize input and output data
        self.x = data1[:, [0, 1]]
        self.input_mean = np.mean(self.x, axis=0)
        self.input_std = np.std(self.x, axis=0)
        self.x = torch.from_numpy((self.x-self.input_mean)/self.input_std)

        self.y = data1[:, 4:]
        self.output_mean = np.mean(self.y, axis=0)
        self.output_std = np.std(self.y, axis=0)
        self.y = torch.from_numpy((self.y-self.output_mean)/self.output_std)

        self.n_samples = data1.shape[0]

    # support indexing such that dataset[i] can
    # be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = lineDataSet()

first_data = dataset[0]
inputs, outputs = first_data
#print(inputs, outputs)
#print(inputs*dataset.input_std+dataset.input_mean,
#      outputs*dataset.output_std+dataset.output_mean)

# Expected Output
# tensor([8.0976e+01, 1.9202e+03, 4.2170e-02, 2.0000e-01]) tensor([364.2700,  78.4050]

#-----Network model-------
class Network(nn.Module):
    def __init__(self, channels=1):  # default grayscale
        super().__init__()
        # self.batch1 = nn.BatchNorm1d(2)
        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.linear2 = nn.Linear(in_features=2, out_features=2)
        # self.linear3 = nn.Linear(in_features=20, out_features=20)
        # self.linear4 = nn.Linear(in_features=2, out_features=2)
        # self.linear5 = nn.Linear(in_features=2, out_features=2)
        # self.linear6 = nn.Linear(in_features=2, out_features=2)
        # self.linear7 = nn.Linear(in_features=2, out_features=2)
        self.out = nn.Linear(in_features=2, out_features=2)

    def forward(self, t):
        # t=self.batch1(t)
        t = self.linear1(t)
        t = torch.relu(t)
        t = self.linear2(t)
        t = torch.relu(t)
        # t = self.linear3(t)
        # t = torch.tanh(t)
        # More layers generate more pronounced seperation
        # t = self.linear2(t)
        # t = F.tanh(t)
        t = self.out(t)
        return t




#-----Plots-----
dataset = lineDataSet()

best_network = Network()
best_network.load_state_dict(torch.load('best_model.pth'))

layer1_output = []
layer2_output = []
  
def hook1(module, input, output):
    layer1_output.append(input)

def hook2(module, input, output):
    layer2_output.append(input)

best_network.linear2.register_forward_hook(hook1)
best_network.out.register_forward_hook(hook2)

# Turn of gradiant
with torch.no_grad():
    best_network.eval()

    network = best_network.cpu()

    pred = network(dataset[:][0])

    a = dataset[:][0]
    a = np.ravel(a.numpy())[:,None]

    b = layer1_output[0][0][:]
    b = np.ravel(b.numpy())[:,None]

    c = layer1_output[0][0][:]
    c = np.ravel(c.numpy())[:,None]

    d=np.hstack((a,b,c))
    print(d.shape)

    plt.clf()
    plt.cla()
    plt.close()

    plt.imshow(d, cmap='copper', interpolation='nearest')
    plt.show() 


