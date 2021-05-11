import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import torchvision

from sklearn.decomposition import PCA

batch_size_test = 1000
batch_size_train = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def assign_weights(self, data_point):
        low = high = 0

        for param in self.parameters():
            high += int(np.product(param.data.shape))
            param.data = torch.from_numpy(data_point[low:high].reshape(param.data.shape)).cuda().float()
            low += int(np.product(param.data.shape))


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size=batch_size_test, shuffle=True, pin_memory=True
)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size=batch_size_train, shuffle=True, pin_memory=True
)


weights = np.load('weights.npy')

network = Net().cuda()

pca = PCA(n_components=2)
pca.fit(weights)
pcs = pca.transform(weights)

pcs = pcs.T

border = 2

x_min = math.floor(min(pcs[0])) - border
print('     x_min: ', x_min)
x_max = math.ceil(max(pcs[0])) + border
print('     x_max: ', x_max)
y_min = math.floor(min(pcs[1])) - border
print('     y_min: ', y_min)
y_max = math.ceil(max(pcs[1])) + border
print('     y_max: ', y_max)
resolution = 3

xs = np.linspace(x_min, x_max, ((x_max - x_min) * resolution) + 1)
ys = np.linspace(y_min, y_max, ((y_max - y_min) * resolution) + 1)
plot_space = np.array([])

np.save('pcs', pcs)
np.save('xs', xs)
np.save('ys', ys)
np.save('resolution', resolution)

for y in ys:
    if plot_space.size > 0:
        plot_space = np.vstack((plot_space, np.array(list(zip(xs, [y]*len(xs))))))
    else:
        plot_space = np.array(list(zip(xs, [y]*len(xs))))

loss_matrix = np.array([])
logged_loss_matrix = np.array([])

train_loss_matrix = np.array([])
train_logged_loss_matrix = np.array([])

i = 0
iterations = plot_space.shape[0]
for plot in plot_space:
    plot_weight = pca.inverse_transform(plot)
    i += 1
    print('     Plot ' + str(i) + ' of ' + str(iterations))
    network.assign_weights(plot_weight)

    loss = 0
    for data, target in test_loader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = network(data)
        loss += F.nll_loss(output, target, reduction='sum').item()

    logged_loss = np.log10(loss + 1e-3)

    train_loss = 0
    for data, target in train_loader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = network(data)
        loss += F.nll_loss(output, target, reduction='sum').item()

    train_logged_loss = np.log10(train_loss + 1e-3)

    if loss_matrix.size > 0:
        loss_matrix = np.append(loss_matrix, loss)
        logged_loss_matrix = np.append(logged_loss_matrix, logged_loss)

        train_loss_matrix = np.append(train_loss_matrix, train_loss)
        train_logged_loss_matrix = np.append(train_logged_loss_matrix, train_logged_loss)
    else:
        loss_matrix = np.array(loss)
        logged_loss_matrix = np.array(logged_loss)

        train_loss_matrix = np.array(train_loss)
        train_logged_loss_matrix = np.array(train_logged_loss)

loss_matrix = loss_matrix.reshape((len(ys), len(xs)))
logged_loss_matrix = logged_loss_matrix.reshape((len(ys), len(xs)))

train_loss_matrix = loss_matrix.reshape((len(ys), len(xs)))
train_logged_loss_matrix = logged_loss_matrix.reshape((len(ys), len(xs)))

np.save('loss_matrix', loss_matrix)
np.save('logged_loss_matrix', logged_loss_matrix)

np.save('train_loss_matrix', train_loss_matrix)
np.save('train_logged_loss_matrix', train_logged_loss_matrix)

pca = PCA()
pca.fit(weights)

np.save('proportion', pca.explained_variance_ratio_)
np.save('cumulative_proportion', np.cumsum(pca.explained_variance_ratio_))
