import torch as t
import torch.nn as nn
import torch.nn.functional as F
import dlc_practical_prologue as dlc

from training import train_rounds
from models import ConvNet, ConvNetWeightSharing

num_samples = 1000
num_epochs = 25
num_rounds = 10

params = {}
params['num_hidden_1'] = 10 # units in the hidden layer

# fix a seed for reproducible results
t.random.manual_seed(6)

# train simple ConvNet
print("Training the Convolutional net.")
print("num_samples: %d, num_epochs: %d, num_rounds: %d, num_hidden_1: %d, loss: CrossEntropyLoss" % (num_samples, num_epochs, num_rounds, params['num_hidden_1']))
train_rounds(num_rounds, ConvNet, num_epochs, nn.CrossEntropyLoss(), params, aux=False)

# train ConvNet with weight sharing
t.random.manual_seed(6)
print("\nTraining the Convolutional net with weight sharing.")
print("num_samples: %d, num_epochs: %d, num_rounds: %d, num_hidden_1: %d, loss: CrossEntropyLoss" % (num_samples, num_epochs, num_rounds, params['num_hidden_1']))
train_rounds(num_rounds, ConvNetWeightSharing, num_epochs, nn.CrossEntropyLoss(), params, aux=False)

# train ConvNet with weight sharing and auxiliary losses
t.random.manual_seed(6)
print("\nTraining the Convolutional net with weight sharing and auxiliary loss.")
print("num_samples: %d, num_epochs: %d, num_rounds: %d, num_hidden_1: %d, loss: CrossEntropyLoss" % (num_samples, num_epochs, num_rounds, params['num_hidden_1']))
train_rounds(num_rounds, ConvNetWeightSharing, num_epochs, nn.CrossEntropyLoss(), params, aux=True)