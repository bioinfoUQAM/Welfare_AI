# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:19:58 2020

@author: Yesmin
"""
import numpy as np
import torch
import argparse
from torch import nn, optim
from YasmineDatasetCC import YKDataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from pathlib import Path
import os


# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)
#     elif type(m) == nn.Conv2d:
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.xavier_uniform_(m.bias)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, (3, 3), padding=2)
        self.conv1_bn = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=2)
        self.conv2_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(32 * 49*2, 120)
        self.dense1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.dense2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (2, 2))
        print(self.num_flat_features(x))
        # x = F.interpolate(x, size=(48,1), mode='bilinear')
        x = x.view(-1, self.num_flat_features(x))
        print(x.size())
        x = F.relu(self.dense1_bn(self.fc1(x)))
        print(x.size())
        x = F.relu(self.dense2_bn(self.fc2(x)))
        # print(x.size())
        x = self.fc3(x)
        # print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--window_size", help="The size of the sliding window", default=192, type=int)
    ap.add_argument("-l", "--dic_path", help="path of the dictionary "
                                             "with generated sample file paths and labels",
                    default=r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\Generated data\labels_dic_side1.csv")
    #ap.add_argument("-f", "--files_folder",
                    #help="The path of the folder that contains the training samples", default="../results/Cows")
    ap.add_argument("-f", "--files_folder",
                    help="The path of the folder that contains the training samples", default="D:\BA_Yasmine_UQAM\Welfare_AI\dataset\Generated data\side2"
    ap.add_argument("-e", "--n_epochs", help="number of epochs", default=1000, type=int)
    ap.add_argument("-b", "--batch_size", help="The batch size", default=16, type=int)
    ap.add_argument("-s", "--save_path", help="The path of the saved model",
                    default=r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\experiments")
    ap.add_argument("-n", "--experiment_name", help="The name of the experiment", default='LeNet', type=str)
    args = vars(ap.parse_args())
    return ap.parse_args()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = 0.001
    test_split = .2
    shuffle_dataset = True
    random_seed = 8
    window_size = args.window_size
    dic_path = args.dic_path
    save_path = Path(args.save_path)
    exp_name = args.experiment_name
    print(f'batch_size{batch_size}')
    print(f'shuffle_dataset{shuffle_dataset}')
    print(f'window_size{window_size}')


    # YKDataset
    dataset = YKDataset(window_size=window_size, dic_path=dic_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler)
    net = LeNet()
    # net.apply(init_weights)
    print(net)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    print(f'criterion{criterion}')
    print(f'optimizer{optimizer}')

    n_total_steps = len(train_loader)
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (files, labels, id) in enumerate(train_loader):
            print('f', files.size())
            # 5*32*194 ( 5 = batch_size)
            # 45* 6208
            # files = files.reshape(-1, 32*window_size).to(device)
            # files = files.reshape(batch_size, 4, window_size, -1).to(device)
            files = files.reshape(-1, 8, 4, window_size).to(device)
            #files = files.to(device)
            # files = files.to(device)

            labels = labels.to(device)
            # print('l', labels.size())
            # forward
            outputs = net(files.float())

            loss = criterion(outputs, labels.long())
            running_loss += loss.item()
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % batch_size == 0:
                print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss {loss.item():.4f}')
        train_losses.append(running_loss / batch_size)  # mean losses of every epoch

    net.eval()

    # test
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for files, labels, ids in test_loader:
            # files = files.to(device)

            files = files.reshape(-1, 8, 4, window_size).to(device)
            print(files.size())
            print(files.size())
            labels = labels.to(device)
            outputs = net(files.float())

            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct = (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'accuracy = {acc} %')
        print('mean train losses', train_losses)

    # torch.save(net.state_dict(), save_path / "model.pth")
    # print("model saved in ", save_path)

    i = 0
    while os.path.exists(os.path.join(save_path, f'{exp_name}{i}.pth')):
        i += 1
    torch.save(net.state_dict(), os.path.join(save_path, f'{exp_name}{i}.pth'))
    print("model saved in ", save_path)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
