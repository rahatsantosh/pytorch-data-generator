import torch
import os
from PIL import Image
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
   'Characterizes a dataset for PyTorch'
   def __init__(self, list_IDs, labels):
       'Initialization'
       self.labels = labels
       self.list_IDs = list_IDs

   def __len__(self):
       'Denotes the total number of samples'
       return len(self.list_IDs)

   def __getitem__(self, index):
       'Generates one sample of data'
       # Select sample
       ID = self.list_IDs[index]

       # Load data and get label
       X = torch.load('data/' + ID + '.pt')
       y = self.labels[ID]

       return X, y


class Dataset2(data.Dataset):
   def __init__(self, root):
       self.path_list = []
       self.label_list = []
       count = 0
       dir_list = [x[1] for x in os.walk(root)][0]
       labels = np.arange(len(dir_list))
       label_dict = {dir_list[i]:labels[i] for i in range(len(dir_list))}
       for a in dir_list:
           d = os.path.join(root,a)
           for path in os.listdir(d):
                if os.path.isfile(os.path.join(d, path)):
                    count += 1
                    self.path_list.append(os.path.join(d, path))
                    self.label_list.append(label_dict[a])
       self.length = count

   def __len__(self):
       return self.length

   def __getitem__(self, index, transform=None):
       img = Image.open(self.path_list[index])
       if transform:
          x = transform(img)
       y = self.label_list[index]

       return x, y
