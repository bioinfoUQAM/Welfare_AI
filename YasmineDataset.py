# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:10:05 2020

@author: Yesmin
"""
from Welfare_AI_dataset import CowsDataset
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import math

DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), 'Dictionary_Kinematics.xlsx')
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'ScaledCoordinates_Post-Trial.xlsx')
#information about the mother dataset( you can change the Side to 'side1' or 'side2' ) 
#in order to determine the size of the sliding  window
dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
dictionary = dictionary.to_numpy()
names = CowsDataset.get_side_sheets('side2')
real_dataset = CowsDataset(names)
window_size = CowsDataset.sliding_window(real_dataset, 'side2')

#normalize the data
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed


class YKDataset(Dataset):
    def __init__(self, transform=normalize):
        #data loading
        dic_path = os.path.join(os.path.dirname(__file__), 'Generated data', 'labels_dic_' + 'side2' + '.csv')
        pathLabels = np.loadtxt(dic_path, delimiter=',', dtype=str, skiprows=1)#load dictionary with file paths and labels
        self.file_paths = pathLabels[:, 1]
        self.indexes = torch.from_numpy(np.array(pathLabels[:, 0], dtype=np.float32))
        self.n_samples = pathLabels.shape[0]
        self.window_size = window_size
        files = [ pd.read_csv(path) for i, path in enumerate(self.file_paths)]#load list of csv files
        files = [pd.DataFrame(file).fillna(0).values for i, file in enumerate(files)]#fill the nan values with 0
        files = [np.array(pd.DataFrame(file).head(self.window_size)) for i, file in enumerate(files)]# window_size=CowsDataset.sliding_window(dataset) const
        self.files = [torch.from_numpy(file) for i, file in enumerate(files)]#convert to torch 
        self.labels = torch.from_numpy(np.array(pathLabels[:, 2], dtype=np.float32))
        self.transform = transform
       

    def __getitem__(self, index):
        #dataset[0]
        #return np.loadtxt(self.file_paths[index], delimiter=',', dtype=str, skiprows=1), self.labels[index]
        file_path = self.file_paths[index]
        file_label = self.labels[index]
        #file = pd.read_csv(file_path)
        file = self.files[index]
         # Normalize your data here
        if self.transform:
            file = self.transform(file)
        # file = pd.DataFrame(file).fillna(0).values#fill the nan values with 0
        # file = pd.DataFrame(file).head(self.window_size)# window_size=CowsDataset.sliding_window(dataset) const
        return file, file_label, file_path
        # return{
        #     'file': file,
        #     'label': file_label,
        #     'file_id': file_path
        # }


    def __len__(self):
        #len(dataset)
        return self.n_samples


dataset = YKDataset()
# first_data = dataset[0]
# features, labels, id = first_data
# print(features, labels, id)
batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True, num_workers=0)
dataiter = iter(dataloader)
data = dataiter.next()
features, labels, id = data
print('features.size() ', features.size())
print('labels.size() ', labels.size())
print(features, labels, id)


#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
for epoch in range(num_epochs):
    for i,(inputs, targets, ids) in enumerate(dataloader):
        #forwardpropagation, backwardpropagation, update weights
        if(i+1) % 4 == 0:
            print(f'epoch {epoch +1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')