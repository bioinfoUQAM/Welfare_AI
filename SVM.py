# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:45:42 2020

@author: Yesmin
"""
from sklearn import svm
import torch
import numpy as np
import argparse
from YasmineDatasetCC import YKDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--window_size", help="The size of the sliding window", default=192, type=int)
    ap.add_argument("-l", "--dic_path", help="path of the dictionary "
                                              "with generated sample file paths and labels",
                    default="../scripts/labels_dic_side2.csv")
    ap.add_argument("-f", "--files_folder",
                    help="The path of the folder that contains the training samples", default="../results/Cows")
    ap.add_argument("-ft", "--files_folder_test",
                    help="The path of the folder that contains the training samples", default="../results/test_dataset/side2")
    ap.add_argument("-lt", "--dic_path_test", help="path of the dictionary "
                                              "with real cows file paths and labels",
                    default="../results/test_dataset/side2/labels_dic_testside2.csv")
    # ap.add_argument("-e", "--n_epochs", help="number of epochs", default=1000, type=int)
    # ap.add_argument("-b", "--batch_size", help="The batch size", default=16, type=int)
    # ap.add_argument("-s", "--save_path", help="The path of the saved model",
    #                 default="../experiments")
    args = vars(ap.parse_args())
    return ap.parse_args()


def SVM():
    #device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_split = .2
    shuffle_dataset = True
    # random_seed = 8
    window_size = args.window_size
    # window_size = 192
    dic_path = args.dic_path
    folder_path = args.files_folder
    dic_path_t = args.dic_path_test
    folder_path_t = args.files_folder_test
    dataset = YKDataset(window_size=window_size, dic_path=dic_path, folder_path=folder_path)
    batch_size = len(dataset)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    
    test_dataset = YKDataset(window_size=window_size, dic_path=dic_path_t, folder_path=folder_path_t)

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    for i,(files,labels,id) in enumerate(train_loader):
        X_train = files.reshape(-1, 32*192).data.numpy()
        y_train = labels.data.numpy()
    for i,(files,labels,id) in enumerate(val_loader):
        X_val = files.reshape(-1, 32*192).data.numpy()
        y_val = labels.data.numpy()
    
    for i,(files,labels,id) in enumerate(test_loader):
        X_test = files.reshape(-1, 32*192).data.numpy()
        y_test = labels.data.numpy()    
         

    clf = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)
    print("Score of the classifier with random train/test split: ", clf.score(X_val, y_val))
    
    y_pred_test = clf.predict(X_test)
    print("Score of the classifier with real cows dataset : ", clf.score(X_test, y_test))

def main():
     args = parse_args()
     SVM(args)
    


if __name__ == "__main__":
    main()

