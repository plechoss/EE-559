import torch as t
import torch.nn as nn
import torch.nn.functional as F
import dlc_practical_prologue as dlc
from models import ConvNet, ConvNetWeightSharing

nb_samples = 1000
epochs = 25
rounds = 10
batch_size = 25

#training a model in batches
def train_net(model, epochs, data_in, labels, classes, criterion, params, aux):
    learning_rate = 1e-3
    net = model(params)
    optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)
    losses = t.Tensor(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_index in range(0, data_in.size(0), batch_size):
            output, aux1, aux2 = net(data_in.narrow(0, batch_index, batch_size))
            
            loss = criterion(output, labels.narrow(0, batch_index, batch_size))
            if aux:
                a = classes.narrow(0, batch_index, batch_size)
                classes1, classes2 = t.chunk(a, chunks=2, dim=1)
                
                loss_aux1 = criterion(aux1, t.squeeze(classes1))
                loss_aux2 = criterion(aux2, t.squeeze(classes2))
                #print("loss_aux1: ")
                #print(loss_aux1)
                #print("loss_aux2: ")
                #print(loss_aux2)
                loss += 0.3*(loss_aux1 + loss_aux2)
            epoch_loss += loss.item()
            losses[epoch] = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #('Epoch loss: {:0.2f}'.format(epoch_loss))
    return net, loss

def evaluate_model(model):
    test_errors = t.Tensor(10)
    for k in range(10):
        #regenerate datasets
        _, _, _, test_input, test_target, _ = dlc.generate_pair_sets(nb_samples)

        test_output, _, _ = model(test_input)
        nb_test_errors = t.sum(test_output.argmax(dim=1)!=(test_target))
        test_errors[k] = nb_test_errors
        print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) // test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    test_errors = 100*test_errors/test_input.size(0)
    print('Test error mean {:0.2f}%'.format(test_errors.mean()))
    print('Test error std {:0.2f}%'.format(test_errors.std()))
