import math
import numpy as np
import pickle
from random import randint

np.random.seed(69)
class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0
        
class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)
    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        try:
            ret = 1 / (1 + math.exp(-x * 1.0))
        except OverflowError:
            ret = float('inf')
        finally:
            return ret
        

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForward(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)
        
    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0

class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None)) # bias neuron
            layer[-1].setOutput(1) # setting output of bias neuron as 1
            self.layers.append(layer)
            
    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err
        
    def feedForward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForward()

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]: #reverse the order
            for neuron in layer:
                neuron.backPropagate()

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop() # removing the bias neuron
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()# removing the bias neuron
        return output
 


def main():
    topology = []
    topology.append(4)
    topology.append(16)
    topology.append(4)
    #net = Network(topology)
    net = pickle.loads(open('net.model','r').read())
    Neuron.eta = 0.09
    Neuron.alpha = 0.01
    inputs = []
    outputs = []
    for i in range(50):
        input = []
        for j in range(4):
            input.append(randint(-1, 10))
        inputs.append(input)
    for inp in inputs:
        best = max(inp)
        idx = inp.index(best)
        output = [0,0,0,0]
        output[idx] = 1
        outputs.append(output)
        
    c = 0
    train = True
    
    while True and train:
        c += 1
        err = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForward()
            net.backPropagate(outputs[i])
            err = err + net.getError(outputs[i])
        if c % 10000 == 0:
            print "error: ", err
            with open('net.model', 'wb') as out:
                out.writelines(pickle.dumps(net))
            out.close()
            
        if err < 0.01:
            break
    with open('net.model', 'wb') as out:
        out.writelines(pickle.dumps(net))
    out.close()
    while True:
        a = input("type input :")
        i = []
        for b in a:
            i.append(b)
        net.setInput(i)
        net.feedForward()
        print net.getThResults() 


if __name__ == '__main__':
    main()

