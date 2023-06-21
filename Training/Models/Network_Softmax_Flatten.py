import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class NetSoftmax(nn.Module):
    def __init__(self, output_size):
        super(NetSoftmax, self).__init__()
        self.fc1 = nn.Linear(196020, 128)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128,output_size)
        self.batchnorm1d = nn.BatchNorm1d(196020)
        xavier_uniform_(self.fc1.weight)
        xavier_uniform_(self.fc2.weight)
        xavier_uniform_(self.fc3.weight)
        xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.batchnorm1d(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, -1

    def predict(self, x):
        x = self.forward(x)
        #probs = torch.softmax(x, dim=1)
        return x

    def activations(self, x):
        # outputs activation this is not necessary just for fun
        z1 = self.conv1(x)
        a1 = torch.relu(z1)
        out = self.pool(a1)

        z2 = self.conv2(out)
        a2 = torch.relu(z2)
        out = self.pool2(a2)
        out = out.view(out.size(0), -1)
        return z1, a1, z2, a2, out