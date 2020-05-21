from torch import Tensor, max
import math

class Optimizer(object):
    '''Class to optimize model parameters using stochastic gradient descent.'''
    def __init__(self, parameters, eta):
        self.params = parameters # model parameters
        self.eta = eta # learning rate
        
    def step(self):
        '''Performs one step of gradient descent on all given parameters.'''
        for (param, gradParam) in self.params:
            param -= self.eta*gradParam

def generate_disc_set(nb):
    '''Generates training set uniformely distributed in [0,1], with label 1 inside
    disk centered at [0.5, 0.5] of radius 1/sqrt(2pi) and label 0 outside.'''
    inp = Tensor(nb, 2).uniform_(0, 1)
    target = inp.sub(0.5).pow(2).sum(1).sub(1./(2*math.pi)).sign().sub(1).div(-2).long()
    return inp, target

def convert_to_one_hot_labels(inp, target):
    '''Convert label vector to one hot label matrix.'''
    tmp = inp.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def compute_stats(model, data_input, data_target, mini_batch_size=100):
    '''Compute accuracy and the number of wrongly predicted instances from test data.accuracy and '''
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))#.reshape((mini_batch_size, 2))
        _, predicted_classes = max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    accuracy = 100 - (100*(nb_data_errors / len(data_target)))
    print('Accuracy : ', accuracy, '%')
    print('Error rate: ', 100*nb_data_errors/len(data_target), '%')
    return

def train_model(model, train_input, train_one_hot_target, criterion, optimizer, nb_epochs=100, mini_batch_size=5, verbose=False):
    '''Train the given model using the given loss function with the given optimizer.'''
    for e in range(0, nb_epochs):
        loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            minibatch_loss = criterion.forward(output, train_one_hot_target.narrow(0, b, mini_batch_size))
            loss += minibatch_loss
            model.zero_grad() # zero all parameter gradients
            model.backward(criterion.backward(train_one_hot_target.narrow(0, b, mini_batch_size)))
            optimizer.step() # train with one step of stochastic gradient descent
        if verbose:
            if e%10 == 0:
                print('Mean loss epoch {} : {:.2f} %'.format(e, 100*loss.item()/train_input.shape[0]))
     