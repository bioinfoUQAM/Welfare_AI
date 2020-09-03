# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:42:43 2020

@author: Yesmin
"""


import numpy as np
import torch
import argparse
from torch import nn, optim
from YasmineDataset2 import YKDataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


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
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (2, 2))
        
        # x = F.interpolate(x, size=(48,1), mode='bilinear')
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.dense1_bn(self.fc1(x)))
        
        x = F.relu(self.dense2_bn(self.fc2(x)))
        # print(x.size())
        #x = self.fc3(x)
        x = self.sigmoid(self.fc3(x))# if BCELoss is used
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
                    default=r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\Generated data\labels_dic_side2.csv")
    ap.add_argument("-f", "--files_folder",
                    help="The path of the folder that contains the training samples", default=r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\Generated data\side2")
    ap.add_argument("-e", "--n_epochs", help="number of epochs", default=100, type=int)
    ap.add_argument("-b", "--batch_size", help="The batch size", default=3, type=int)
    ap.add_argument("-s", "--save_path", help="The path of the saved model",
                    default="../experiments")
    args = vars(ap.parse_args())
    return ap.parse_args()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = parse_args().n_epochs
    batch_size = parse_args().batch_size
    learning_rate = 0.001
    test_split = .2
    shuffle_dataset = True
    random_seed = 8
    window_size = parse_args().window_size
    dic_path = parse_args().dic_path
    save_path = Path(parse_args().save_path)
    folder_path = parse_args().files_folder
    
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
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    
    n_total_steps = len(train_loader)
    train_losses = []
    n_correct = 0
    n_samples = 0
    loss_values = []
    for epoch in tqdm(range(num_epochs)):
        preds = []
        targets = []
        running_loss = 0.0
        for i, (files, labels, id) in enumerate(train_loader):
            files = files.reshape(-1, 8, 4, window_size).to(device)
    
            labels = labels.to(device)
            
            # forward
            outputs = net(files.float())
    
    
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #accuracy
            predictions = torch.round(outputs)
            target = labels.float()
            preds.append(predictions.data)
            targets.append(labels.data)
            n_samples += labels.shape[0]
            
                
   
        
    
        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Visualization data
        if step % 10 == 0:
            vis.plot_loss(np.mean(loss_values), step)
            loss_values.clear()
        #n_correct += (predictions == labels).sum().item()
        
    preds = torch.cat(preds)
    
    
    targets = torch.cat(targets)
    
    # acc = 100.0 * n_correct / n_samples
    print("Accuracy on train set is" , accuracy_score(targets,preds))
    print( confusion_matrix(targets,preds))
    print( classification_report(targets,preds))    
        
    # print(f'train accuracy = {acc} %')
    
            # (i + 1) % batch_size == 0:
                #print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss {loss.item():.4f}')
        # train_losses.append(running_loss / batch_size)  # mean losses of every epoch
    
    net.eval()
        
    
    
    # test
    with torch.no_grad():
        preds = []
        targets = []
        for i, (files, labels, ids) in enumerate(test_loader):
    
            files = files.reshape(-1, 8, 4, window_size).to(device)
            labels = labels.to(device)
            outputs = net(files.float())
            # value, index
            predictions = torch.round(outputs)
            target = labels.float()
            preds.append(predictions.data)
            targets.append(labels.data)
            
    
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        print("Accuracy on test set is" accuracy_score(targets,preds))
        print( confusion_matrix(targets,preds))
        print( classification_report(targets,preds))
        
    
    torch.save(net.state_dict(), save_path / "model.pth")
    print("model saved in ", save_path)


class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
