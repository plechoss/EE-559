from torch import random, set_grad_enabled
from modules import *
from helpersAndTrain import *

set_grad_enabled(False) # turn autograd globally off
random.manual_seed(3) # set seed for reproducible results

# generate data
train_inp, train_target = generate_disc_set(1000)
test_inp, test_target = generate_disc_set(1000)

# make variance = 1, mean = 0
train_inp = train_inp.sub(train_inp.mean()).div(train_inp.std())
test_inp = test_inp.sub(test_inp.mean()).div(test_inp.std())
train_target_hot = convert_to_one_hot_labels(train_inp, train_target)

# network with two input units, two output units, three hidden layers of 25 units
model = Sequential(Linear(2, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 2))
optim = Optimizer(model.param(), 0.001)
loss = LossMSE()

# train for 100 epochs
train_model(model, train_inp, train_target_hot, loss, optim, mini_batch_size=100, nb_epochs=100, verbose=True)

# print stats
print('\nTraining set')
compute_stats(model, train_inp, train_target, mini_batch_size=1)
print('\nTest set')
compute_stats(model, test_inp, test_target, mini_batch_size=1)