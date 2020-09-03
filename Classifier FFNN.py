import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from YasmineDataset import YKDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size_side1 = 6208 #(194*32)
input_size_side2 = 6144 #(192*32)
hidden_size = 100
num_classes = 2
num_epochs = 2
batch_size = 5
learning_rate = 0.001
test_split = .2
shuffle_dataset = True
random_seed = 8


#YKDataset
dataset = YKDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True, num_workers=0)


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
dataiter = iter(dataloader)
data = dataiter.next()
features, labels, id = data
print('features.size() ', features.size())
print('features.size() ', labels.size())
# print('labels.size() ', labels.size())
# print(features, labels, id)


class NeuralNet(nn.Module):
    def __init__(self, input_size_side1, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size_side1, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size_side2, hidden_size, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(files, labels, id) in enumerate(train_loader):
        #5*32*194 ( 5 = batch_size)
        # 45* 6208
        files = files.reshape(-1, 32*192).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(files.float())
        loss = criterion(outputs, labels.long())

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%5 == 0:
            print(f'epoch {epoch +1}/{num_epochs}, step {i+1}/{n_total_steps}, loss {loss.item():.4f}')


#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for files, labels, ids in test_loader:
        files = files.reshape(-1, 32*192).to(device)
        labels = labels.to(device)
        outputs = model(files.float())

       #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()

    acc = 100.0*n_correct/ n_samples
    print(f'accuracy = {acc} %')





