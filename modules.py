from torch import Tensor, max
import math

class Module(object):
    '''Module superclass from which all layers will inherit.'''
    def __init__(self):
        '''Constructor of the Module class.'''
        # attributes needed for all modules
        self.output = Tensor() # output of module (after calling forward method)
        self.gradInput = Tensor() # gradient with respect to input to model (result of backprop)
        self.type = str()
        
    def __call__(self, *inp, **kwargs):
        '''Makes layer callable like a function and directly returns the result of forward().'''
        return self.forward(*inp, **kwargs)
   
    def forward(self, *inp, **kwargs):
        '''should get for input, and returns, a tensor or a tuple of tensors.'''
        return self.output
        
    def backward(self, *gradwrtoutput):
        '''should get as input a tensor or a tuple of tensors containing the gradient of the loss
with respect to the module's output, accumulate the gradient wrt the parameters, and return a
tensor or a tuple of tensors containing the gradient of the loss wrt the module's input.'''
        return self.gradInput
    
    def zero_grad(self):
        '''Sets all the gradients to zero.'''
        self.gradInput *= 0.
        if hasattr(self, 'weights'):
            self.gradWeights *= 0.
        if hasattr(self, 'biases'):
            self.gradBiases *= 0.
        
    def param(self):
        '''Return a list of pairs, each composed of a parameter tensor, and a gradient tensor
            of same size. This list should be empty for parameterless modules (e.g. ReLU)'''
        if hasattr(self, 'weights') and hasattr(self, 'biases'):
            return [(self.weights, self.gradWeights), (self.biases, self.gradBiases)]
        elif hasattr(self, 'weights'):
            return [(self.weights, self.gradWeights)]
        elif hasattr(self, 'biases'):
            return [(self.biases, self.gradBiases)]
        else : return []

class Linear(Module):
    '''Module to implement fully connected layers of arbitrary size.'''
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.inp = Tensor()
        self.weights = Tensor(out_dim,in_dim).normal_() # BIG CHANGE
        self.biases = Tensor(out_dim,1).normal_()
        self.gradWeights = Tensor(self.weights.size()).zero_()
        self.gradBiases = Tensor(self.biases.size()).zero_()
        self.type = 'Linear'
           
    def forward(self, inp):
        self.inp = inp
        self.output = (self.weights.mm(inp.t()) + self.biases).t()
        return self.output
        
    def backward(self, gradOutput):
        self.gradInput =  (self.weights.t().mm(gradOutput.t())).t()
        gradWeights = Tensor.mm(gradOutput.t(), self.inp)
        gradBiases = gradOutput.t()
        # sum the gradients for the weights and biases
        self.gradWeights += gradWeights
        self.gradBiases += gradBiases.sum(1).unsqueeze(1) # unsqueeze returns vector
        return self.gradInput
        
class Tanh(Module):
    '''Module to implement Tanh acivation module.'''
    def __init__(self):
        super(Tanh, self).__init__()
        self.type = 'Tanh'
        self.inp = Tensor()
        
    def forward(self, inp):
        self.inp = inp
        self.output = inp.tanh()
        return self.output
        
    def backward(self, gradOutput):
        # derivative of Tanh on input
        dtanh = (1 - self.inp.tanh().pow(2))
        return gradOutput.mul(dtanh)
    
class ReLU(Module):
    '''Rectified linear unit activation module.'''
    def __init__(self):
        super(ReLU, self).__init__()
        self.inp = Tensor()
        self.type = 'ReLU'
    
    def forward(self, inp):
        self.inp = inp.clone()
        inp[inp < 0] = 0
        self.output = inp.clone()
        return self.output
        
    def backward(self, gradOutput):
        step = self.inp.clone()
        # derivative of ReLU is step function applied to original input
        step[step > 0] = 1
        step[step < 0] = 0
        self.gradInput = gradOutput.mul(step)
        return self.gradInput
    
class Sequential(Module):    
    '''Container to store several layers sequentially.'''
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.modules = []
        self.size = 0
        self.type = 'Sequential container'
        for arg in args:
            self.add(arg)
        print(self)

    def __str__(self):
        '''Display content of sequential container.'''
        string = 'New neural net\n'
        for ind, module in enumerate(self.modules):
            if module.type == 'Linear':
                string += '   Module ' + str(ind) + ': ' + module.type +', ' + str(module.weights.shape) + '\n'
            else:
                string += '   Module ' + str(ind) + ': ' + module.type + '\n'
        return string
    
    def add(self, module, index=None):
        '''Add new layer at position index. By default is added as new last layer.'''
        if index == None: index = self.size
        if index < 0 or index > self.size:
            raise ValueError('Supplied index is out of range for number of modules in this sequence.')
        self.modules.insert(index, module)
        self.size += 1
        
    def forward(self, inp):
        temp = inp.clone()
        for module in self.modules:
            temp = module(temp) # feed forward loop
        return temp
        
    def backward(self, gradOutput):
        temp = gradOutput.clone()
        for module in reversed(self.modules):
            temp = module.backward(temp) # backpropagation loop
        return temp

    def param(self): 
        '''Returns a flattened list of each module's parameters with a tuple of the
        actual parameter and its gradient.'''
        return [ p for module in self.modules for p in module.param() ]
    
    def zero_grad(self):
        '''Set the gradient of each parameter in all the modules to zero.'''
        for module in self.modules:
            module.zero_grad()
            
class LossMSE(Module):
    '''Module to implement mean square loss.'''
    def __init__(self):
        super(LossMSE, self).__init__()
        self.inp = Tensor()
        self.type = 'MSE loss'
        
    def forward(self, inp, targets):
        self.inp = inp.clone()
        self.output = (inp - targets).pow(2).sum()
        return self.output
        
    def backward(self, targets):
        self.gradInput = 2. * (self.inp - targets)
        return self.gradInput