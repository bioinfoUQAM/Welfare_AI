# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:45:42 2020

@author: Yesmin
"""
from sklearn import svm
import torch
import argparse
import numpy as np
from YasmineDataset_Mat import YKDataset
from YasmineDatasetCC import TestDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--window_size", default=192, type=int,
                    help="The size of the sliding window")

    # ap.add_argument("-l", "--dic_path", default="../scripts/labels_dic_side2.csv",
    #                 help="path of the dictionary with generated sample file paths and labels" )

    ap.add_argument("-f", "--files_folder", default=r"D:\BA_Yasmine_UQAM\Welfare_AI\YK_dataset2%.mat",
                    help="The path of the folder that contains the training samples", )

    ap.add_argument("-ft", "--files_folder_test", default=r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\test_dataset\side2",
                    help="The path of the folder that contains the training samples")

    ap.add_argument("-lt", "--dic_path_test", default=r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\test_dataset\side2"
                                                      r"\labels_dic_testside2.csv",
                    help="path of the dictionary with real cows file paths and labels")

    ap.add_argument("-nf", "--n_folds", default=10, type=int,
                    help='number of folds for the cross validation')

    ap.add_argument("-v", "--val_folder_path", default="D:\BA_Yasmine_UQAM\Welfare_AI\SVMs",
                    help="The path of the folder that contains the score, confusion matrix and classification report "
                         "of the classifier")

    ap.add_argument("-e", "--eval_folder_path", default="D:\BA_Yasmine_UQAM\Welfare_AI\SVMs",
                    help="The path of the folder that contains the score, confusion matrix and classification report "
                         "of the classifier")

    return ap.parse_args()


def SVM(args):
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shuffle_dataset = True
    # random_seed = 8
    window_size = args.window_size
    # window_size = 192
    # dic_path = args.dic_path
    folder_path = args.files_folder
    dic_path_t = args.dic_path_test
    folder_path_t = args.files_folder_test
    # dic_path = r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\Generated data\labels_dic_side2.csv"
    # folder_path = r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\Generated data\side2"
    # dic_path_t = r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\test_dataset\side2\labels_dic_testside2.csv"
    # folder_path_t = r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\test_dataset\side2"
    # YKDataset
    # dataset = YKDataset(window_size=window_size, dic_path=dic_path, folder_path=folder_path)
    dataset = YKDataset(window_size=window_size, folder_path=folder_path)

    batch_size = len(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    # test_dataset = YKDataset(window_size=window_size, dic_path=dic_path_t, folder_path=folder_path_t)
    test_dataset = TestDataset(window_size=window_size, dic_path=dic_path_t, folder_path=folder_path_t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    kf = KFold(n_splits=args.n_folds, shuffle=True)
    n_fold = 1

    for train_indices, val_indices in kf.split(indices):
        validation_file = Path(args.val_folder_path) / 'linear' / f'{n_fold}val_file2%.csv'
        with open(validation_file, 'w') as file:
            file.write(f"SVM model evaluation on validation set \n")
        #print(n_fold, ' training indices', train_indices)
        #print(n_fold, ' validation indices', val_indices)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        # Creating PT data samplers and loaders:

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=val_sampler)

        for i, (files, labels, id) in enumerate(train_loader):
            X_train = files.reshape(-1, 32 * 192).data.numpy()
            y_train = np.ravel(labels.data.numpy())
        for i, (files, labels, id) in enumerate(val_loader):
            X_val = files.reshape(-1, 32 * 192).data.numpy()
            y_val = np.ravel(labels.data.numpy())

        clf = svm.SVC(kernel='linear')  # Linear Kernel

        # Train the model using the training sets

        clf.fit(X_train, y_train)

        y_pred_val = clf.predict(X_val)
        print("Score of the classifier ", n_fold, " with random train/test split: ", clf.score(X_val, y_val))


        with open(validation_file, 'a') as file:
            file.write(f"{n_fold} , Score of the classifier : {clf.score(X_val, y_val)} \n")
            file.write(f"{n_fold} , confusion matrix:\n {confusion_matrix(y_val, y_pred_val)} \n")
            file.write(f"{n_fold} , classification_report:\n {classification_report(y_val, y_pred_val)}\n")


    # test

        for i, (files, labels, id) in enumerate(test_loader):
            X_test = files.reshape(-1, 32 * 192).data.numpy()
            y_test = labels.data.numpy()
        y_pred_test = clf.predict(X_test)

        evaluation_file = Path(args.eval_folder_path) / 'linear' / f'{n_fold}eval_file2%.csv'
        with open(evaluation_file, 'w') as file:
            file.write("SVM model evaluation on real cows dataset \n")
        with open(evaluation_file, 'a') as file:
            file.write(f" Score of the classifier : {clf.score(X_test, y_test)} \n")
            file.write(f"confusion matrix:\n {confusion_matrix(y_test, y_pred_test)} \n")
            file.write(f"classification_report:\n {classification_report(y_test, y_pred_test)}\n")
        n_fold += 1

def main():
    args = parse_args()
    SVM(args)


if __name__ == "__main__":
    main()
