import torch as t
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
'''Basic convolutional neural network structure with 2 convolution layers and 3 fully connected layers. '''
    def __init__(self, params):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(2, 20, 3)
        self.conv2 = nn.Conv2d(20, 10, 3)

        self.fc1 = nn.Linear(10 * 2 * 2, params['num_hidden_1'])
        self.fc2 = nn.Linear(params['num_hidden_1'], 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x, kernel_size=3):
        # 2x14x14 -> 20x6x6
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x,2,2))
        # 20x6x6 -> 10x2x2
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x,2,2))
        # 10x2x2 -> 40
        x = x.view(-1, 10 * 2 * 2)

        # 40 -> num_hidden_1
        x = F.relu(self.fc1(x))
        # num_hidden_1 -> 20
        x = F.relu(self.fc2(x))
        # 20 -> 2
        x = t.sigmoid(self.fc3(x))

        return x, 0, 0

class ConvNetWeightSharing(nn.Module):

    def __init__(self, params):
        super(ConvNetWeightSharing, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv2d(20, 10, 3)

        self.fc1 = nn.Linear(10 * 2 * 2, params['num_hidden_1'])
        self.fc2 = nn.Linear(params['num_hidden_1'], 10)
        self.fc3 = nn.Linear(20, 2)

    def forward_helper(self, x):
        '''Helper function that performs the individual forward passes of the ConvNet. '''
        # 1x14x14 -> 20x6x6
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x,2,2))
        # 20x6x6 -> 10x2x2
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x,2,2))
        # 10x2x2 -> 40
        x = x.view(-1, 10 * 2 * 2)

        # 40 -> num_hidden_1
        x = F.relu(self.fc1(x))
        # num_hidden_1 -> 20
        x = self.fc2(x)
        return x

    def forward(self, x, kernel_size=3):
        aux1, aux2 = t.chunk(x, chunks=2, dim=1)

        aux1 = self.forward_helper(aux1)
        aux2 = self.forward_helper(aux2)

        x = t.cat((aux1, aux2), dim=1)

        x = t.sigmoid(self.fc3(F.relu(x)))

        return x, t.sigmoid(aux1), t.sigmoid(aux2)
