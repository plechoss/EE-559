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
def train_net(model, epochs, data_in, labels, classes, test_data_in, test_labels, test_classes, criterion, params, aux):
    learning_rate = 1e-3
    net = model(params)
    optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)
    losses = t.Tensor(epochs)

    train_errors = t.Tensor(epochs)
    test_errors = t.Tensor(epochs)
    
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
    
        #out, _, _ = net(data_in)
        #nb_train_errors = t.sum(out.argmax(dim=1)!=(labels))
        #train_errors[epoch] = nb_train_errors
            
        #test_output, _, _ = net(test_data_in)
        #nb_test_errors = t.sum(test_output.argmax(dim=1)!=(test_labels))
        #test_errors[epoch] = nb_test_errors
        #('Epoch loss: {:0.2f}'.format(epoch_loss))
    return net, loss, train_errors, test_errors

def train_rounds(num_rounds, model, epochs, criterion, params, aux, verbose=True):
    train_errors = t.Tensor(num_rounds)
    test_errors = t.Tensor(num_rounds)
    
    for i in range(num_rounds):
        #print('round: ')
        #print(i)
        train_input, train_target, train_classes, test_input, test_target, test_classes = dlc.generate_pair_sets(nb_samples)
        
        net, loss, trains, tests = train_net(model, epochs, train_input, train_target, train_classes, test_input, test_target, test_classes, criterion, params, aux)
        #print("trained")
        #plt.plot(100*tests/data_in.size(0), label='testing error')
        #plt.plot(100*trains/data_in.size(0), label='training error')
        #plt.legend()
        #plt.show()
        
        test_output, _, _ = net(test_input)
        
        nb_test_errors = t.sum(test_output.argmax(dim=1)!=(test_target))
        test_errors[i] = nb_test_errors
        if verbose:
            print('round {:d} - test error Net {:0.2f}% {:d}/{:d}'.format(i, (100 * nb_test_errors) // test_input.size(0), nb_test_errors, test_input.size(0)))
    test_errors = 100*test_errors/train_input.size(0)
    print('Test error mean {:0.2f}%'.format(test_errors.mean()))
    print('Test error std {:0.2f}%'.format(test_errors.std()))

def evaluate_model(model, verbose=True):
    test_errors = t.Tensor(10)
    for k in range(10):
        #regenerate datasets
        _, _, _, test_input, test_target, _ = dlc.generate_pair_sets(nb_samples)

        test_output, _, _ = model(test_input)
        nb_test_errors = t.sum(test_output.argmax(dim=1)!=(test_target))
        test_errors[k] = nb_test_errors
        if verbose:
            print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) // test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    test_errors = 100*test_errors/test_input.size(0)
    print('Test error mean {:0.2f}%'.format(test_errors.mean()))
    print('Test error std {:0.2f}%'.format(test_errors.std()))
