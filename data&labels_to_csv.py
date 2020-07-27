# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 11:17:08 2020

@author: Yesmin
"""


from Welfare_AI_dataset import CowsDataset
import pandas as pd
import os
from Data_generation import GeneratedData
from tqdm import tqdm
import numpy as np
from Clustering_dataset import Hierarchical_clustering



DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), 'Dictionary_Kinematics.xlsx')
side = 'side2' #or 'side2'
n_samples = 5#number of generated cows
n_cows = 9

dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
dictionary = dictionary.to_numpy()
names = CowsDataset.get_side_sheets(side)
dataset = CowsDataset(names)
columns = CowsDataset.get_joint_names(dataset)

generated_dataset=[]
new_dataset = []#list of the new generated dataframes of the generated cows
new_sheets = [] #list of the names of the generated cows
new_targets = []#list of the labels of the generated cows
paths = []# list of the paths of csv generated cow files
#generate n_samples of every cow
for i, sheet in enumerate(tqdm(dataset)):
    new_dataset.append(GeneratedData.create_generated_cow_data(dataset, i, n_samples)[0])
    new_sheets.append(GeneratedData.create_generated_cow_data(dataset, i, n_samples)[1])
    new_targets.append(GeneratedData.create_generated_cow_data(dataset, i, n_samples)[2])
# np.array(new_targets).reshape([n_samples*n_cows,-1])
# save the generated samples in \Generated data\side2
for i, cows in enumerate(new_dataset):
    for j, sample in enumerate(cows):
        # print (sample)
        samples_PATH = os.path.join(os.path.dirname(__file__), 'Generated data', side,  new_sheets[i][j]+side+'.csv')
        paths.append(samples_PATH)
        sample.to_csv(samples_PATH, index=False)
new_targets = Hierarchical_clustering.ground_truth_labels(new_targets, n_samples)#repeat every label n_samples times so that we have the truth labels of every generated data

#create a dictionary with file path and label        
d = {'file path':paths,'label':new_targets} 
labels_dic = pd.DataFrame(d) #dictionary to dataframe 
dic_path =  os.path.join(os.path.dirname(__file__), 'Generated data', 'labels_dic_'+ side +'.csv')
labels_dic.to_csv(dic_path)    
