#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import datasets

from model import Generator, Discriminator
from train import train_DCGAN

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


# In[3]:


import numpy as np
with open("numpy_data.npy","rb") as f:
    pixels = np.load(f)


# In[4]:


pixels = np.resize(pixels,(1201,1,128,128))


# In[5]:


pixels.shape


# In[13]:


# trans = transforms.Compose([
#             transforms.Scale(128),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, ), (0.5, )),
#         ])
# path = r'C:\Users\Monalisha\Desktop\generative modelling\project'
# dataset = datasets.ImageFolder(path, transform=trans)


# In[6]:


dataset = torch.from_numpy(pixels)


# In[9]:


dataset.size()


# In[12]:


g_lr = 0.0002
d_lr = 0.0002
batch_size = 128
num_epochs = 10


# In[13]:



#train_set = MNIST(root='.', train=True, transform=trans, download=True)
train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[22]:




G = Generator().to(device)
D = Discriminator().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

loss_f = nn.BCELoss()

train_DCGAN(G, D, optim_G, optim_D, loss_f, train_loader, num_epochs, device)


# In[ ]:





# 
