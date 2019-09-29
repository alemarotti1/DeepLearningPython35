import os
import datetime as date

dirCin="E:\\Users\\amft\\Documents\\DeepLearningPython35"
dirHome = "C:\\Users\\jeong\\Documents\\Python Scripts\\DeepLearningPython35"
dirLuiz = ""
os.chdir(dirHome)

# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def write_file(list, name):
    f= open(name+".csv","w+")
    for i in list:
        for j in i:
            f.write(str(j)+"; ")
        f.write("\n")
import network

totalResult = []


hidden_layer_maxsize = 17 #increasing one layer at a time
mini_batch_high_times = 10 #increasing 2 items at a time
epoch_high_times = 15  #increasing 5 epochs at a time
eta_change_times = 15 #increasing 0.02 at a time

best_accuracy = 0.0
best_topology = []
best_batch_size = 10
best_epoch_size = 40
best_eta = 0.03

mini_batch_size = 30
epochs=40
eta=0.05
t_data, validation_data, t_data = load_data_wrapper()
out_data = list(t_data)

#%%
#testando diferentes composições de topologia
for i in range(0, hidden_layer_maxsize): 
    topology = [784]
    for j in range(0,i):
        topology.append(50)
    topology.append(10)
    net = network.Network(topology)
    
    print("inicio do treinamento")
    startTime = date.datetime.now()
    net.SGD(training_data=out_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=eta, test_data=validation_data)
    endTime = date.datetime.now()
    print("fim do treinamento")
    print("tempo total de treinamento"+str(endTime-startTime))
    
    
    t_data, validation_data, t_data = load_data_wrapper()
    final_test_data = list(t_data)
    accuracy = np.zeros((11,1))
    hits = np.zeros((11,1))
    total = np.zeros((11,1))
    n_test = len(final_test_data)
    #result = net.evaluate(t_data)
    test_results = [(np.argmax(net.feedforward(x)), y)
                            for (x, y) in final_test_data]
    
    
    for (x, y) in test_results:
        total[y] = total[y] +1
        total[10] = total[10] +1
        if (int(x == y)):
            hits[y] = hits[y] + 1
            hits[10] = hits[10] + 1
    for i in range(0,11):
        accuracy[i] = hits[i]/total[i]
            
    if final_test_data:
        print("result: {} / {}".format(net.evaluate(final_test_data),n_test));
    
    
    resultByTest = []

    print("final result: ")
    print("mini_batch_size: "+str(mini_batch_size))
    resultByTest.append(str(mini_batch_size))
    print("topology: "+str(topology))
    resultByTest.append(str(topology))
    print("epochs: "+str(epochs))
    resultByTest.append(str(epochs))
    print("eta: "+str(eta))
    resultByTest.append(str(eta))
    print("general accuracy: "+str(accuracy[10]))
    resultByTest.append(str(accuracy[10]))
    print("accuracy 0: "+str(accuracy[0]))
    resultByTest.append(str(accuracy[0]))
    print("accuracy 1: "+str(accuracy[1]))
    resultByTest.append(str(accuracy[1]))
    print("accuracy 2: "+str(accuracy[2]))
    resultByTest.append(str(accuracy[2]))
    print("accuracy 3: "+str(accuracy[3]))
    resultByTest.append(str(accuracy[3]))
    print("accuracy 4: "+str(accuracy[4]))
    resultByTest.append(str(accuracy[4]))
    print("accuracy 5: "+str(accuracy[5]))
    resultByTest.append(str(accuracy[5]))
    print("accuracy 6: "+str(accuracy[6]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 7: "+str(accuracy[7]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 8: "+str(accuracy[8]))
    resultByTest.append(str(accuracy[8]))
    print("accuracy 9: "+str(accuracy[9]))
    resultByTest.append(str(accuracy[9]))
    resultByTest.append(str(endTime-startTime))
    if (hits[10]/total[10]) > best_accuracy:
        best_accuracy = hits[10]/total[10]
        best_topology = topology
    totalResult.append(resultByTest)
    
write_file(totalResult, "1ChangingTopology")


#%%

temp_counter = 0

#testando diferentes composições de mini_batch_size
while temp_counter != mini_batch_high_times:
    net = network.Network(best_topology)
    
    #atualizando o tamanho do batch
    mini_batch_size = mini_batch_size + temp_counter*2
    print("inicio do treinamento")
    startTime = date.datetime.now()
    net.SGD(training_data=out_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=eta, test_data=validation_data)
    endTime = date.datetime.now()
    print("fim do treinamento")
    print("tempo total de treinamento"+str(endTime-startTime))
    
    
    t_data, validation_data, t_data = load_data_wrapper()
    final_test_data = list(t_data)
    accuracy = np.zeros((11,1))
    hits = np.zeros((11,1))
    total = np.zeros((11,1))
    n_test = len(final_test_data)
    #result = net.evaluate(t_data)
    test_results = [(np.argmax(net.feedforward(x)), y)
                            for (x, y) in final_test_data]
    
    
    for (x, y) in test_results:
        total[y] = total[y] +1
        total[10] = total[10] +1
        if (int(x == y)):
            hits[y] = hits[y] + 1
            hits[10] = hits[10] + 1
    for i in range(0,11):
        accuracy[i] = hits[i]/total[i]
            
    if final_test_data:
        print("result: {} / {}".format(net.evaluate(final_test_data),n_test));
    
    
    resultByTest = []

    print("final result: ")
    print("mini_batch_size: "+str(mini_batch_size))
    resultByTest.append(str(mini_batch_size))
    print("topology: "+str(topology))
    resultByTest.append(str(topology))
    print("epochs: "+str(epochs))
    resultByTest.append(str(epochs))
    print("eta: "+str(eta))
    resultByTest.append(str(eta))
    print("general accuracy: "+str(accuracy[10]))
    resultByTest.append(str(accuracy[10]))
    print("accuracy 0: "+str(accuracy[0]))
    resultByTest.append(str(accuracy[0]))
    print("accuracy 1: "+str(accuracy[1]))
    resultByTest.append(str(accuracy[1]))
    print("accuracy 2: "+str(accuracy[2]))
    resultByTest.append(str(accuracy[2]))
    print("accuracy 3: "+str(accuracy[3]))
    resultByTest.append(str(accuracy[3]))
    print("accuracy 4: "+str(accuracy[4]))
    resultByTest.append(str(accuracy[4]))
    print("accuracy 5: "+str(accuracy[5]))
    resultByTest.append(str(accuracy[5]))
    print("accuracy 6: "+str(accuracy[6]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 7: "+str(accuracy[7]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 8: "+str(accuracy[8]))
    resultByTest.append(str(accuracy[8]))
    print("accuracy 9: "+str(accuracy[9]))
    resultByTest.append(str(accuracy[9]))
    resultByTest.append(str(endTime-startTime))
    if (hits[10]/total[10]) > best_accuracy:
        best_accuracy = hits[10]/total[10]
        best_batch_size = mini_batch_size
    totalResult.append(resultByTest)
    #aumentando
    temp_counter = temp_counter + 1
    
write_file(totalResult, "2ChangingBatch_size")


#%%

temp_counter = 0

#testando diferentes composições de epochs
while temp_counter != epoch_high_times:
    net = network.Network(best_topology)
    
    
    epochs = epochs + temp_counter*5
    print("inicio do treinamento")
    startTime = date.datetime.now()
    net.SGD(training_data=out_data, epochs=epochs, mini_batch_size=best_batch_size, eta=eta, test_data=validation_data)
    endTime = date.datetime.now()
    print("fim do treinamento")
    print("tempo total de treinamento"+str(endTime-startTime))
    
    
    t_data, validation_data, t_data = load_data_wrapper()
    final_test_data = list(t_data)
    accuracy = np.zeros((11,1))
    hits = np.zeros((11,1))
    total = np.zeros((11,1))
    n_test = len(final_test_data)
    #result = net.evaluate(t_data)
    test_results = [(np.argmax(net.feedforward(x)), y)
                            for (x, y) in final_test_data]
    
    
    for (x, y) in test_results:
        total[y] = total[y] +1
        total[10] = total[10] +1
        if (int(x == y)):
            hits[y] = hits[y] + 1
            hits[10] = hits[10] + 1
    for i in range(0,11):
        accuracy[i] = hits[i]/total[i]
            
    if final_test_data:
        print("result: {} / {}".format(net.evaluate(final_test_data),n_test));
    
    
    resultByTest = []

    print("final result: ")
    print("mini_batch_size: "+str(mini_batch_size))
    resultByTest.append(str(mini_batch_size))
    print("topology: "+str(topology))
    resultByTest.append(str(topology))
    print("epochs: "+str(epochs))
    resultByTest.append(str(epochs))
    print("eta: "+str(eta))
    resultByTest.append(str(eta))
    print("general accuracy: "+str(accuracy[10]))
    resultByTest.append(str(accuracy[10]))
    print("accuracy 0: "+str(accuracy[0]))
    resultByTest.append(str(accuracy[0]))
    print("accuracy 1: "+str(accuracy[1]))
    resultByTest.append(str(accuracy[1]))
    print("accuracy 2: "+str(accuracy[2]))
    resultByTest.append(str(accuracy[2]))
    print("accuracy 3: "+str(accuracy[3]))
    resultByTest.append(str(accuracy[3]))
    print("accuracy 4: "+str(accuracy[4]))
    resultByTest.append(str(accuracy[4]))
    print("accuracy 5: "+str(accuracy[5]))
    resultByTest.append(str(accuracy[5]))
    print("accuracy 6: "+str(accuracy[6]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 7: "+str(accuracy[7]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 8: "+str(accuracy[8]))
    resultByTest.append(str(accuracy[8]))
    print("accuracy 9: "+str(accuracy[9]))
    resultByTest.append(str(accuracy[9]))
    resultByTest.append(str(endTime-startTime))
    if (hits[10]/total[10]) > best_accuracy:
        best_accuracy = hits[10]/total[10]
        best_epoch_size = epochs
    totalResult.append(resultByTest)
    #aumentando
    temp_counter = temp_counter + 1
    
write_file(totalResult, "3Changing_epoch_size")

#%%

temp_counter = 0

#testando diferentes composições de eta
while temp_counter != eta_change_times:
    net = network.Network(best_topology)
    
    #modificanto o eta
    eta = eta + temp_counter*0.02
    
    
    print("inicio do treinamento")
    startTime = date.datetime.now()
    net.SGD(training_data=out_data, epochs=best_epoch_size, mini_batch_size=best_batch_size, eta=eta, test_data=validation_data)
    endTime = date.datetime.now()
    print("fim do treinamento")
    print("tempo total de treinamento"+str(endTime-startTime))
    
    
    t_data, validation_data, t_data = load_data_wrapper()
    final_test_data = list(t_data)
    accuracy = np.zeros((11,1))
    hits = np.zeros((11,1))
    total = np.zeros((11,1))
    n_test = len(final_test_data)
    #result = net.evaluate(t_data)
    test_results = [(np.argmax(net.feedforward(x)), y)
                            for (x, y) in final_test_data]
    
    
    for (x, y) in test_results:
        total[y] = total[y] +1
        total[10] = total[10] +1
        if (int(x == y)):
            hits[y] = hits[y] + 1
            hits[10] = hits[10] + 1
    for i in range(0,11):
        accuracy[i] = hits[i]/total[i]
            
    if final_test_data:
        print("result: {} / {}".format(net.evaluate(final_test_data),n_test));
    
    
    resultByTest = []

    print("final result: ")
    print("mini_batch_size: "+str(mini_batch_size))
    resultByTest.append(str(mini_batch_size))
    print("topology: "+str(topology))
    resultByTest.append(str(topology))
    print("epochs: "+str(epochs))
    resultByTest.append(str(epochs))
    print("eta: "+str(eta))
    resultByTest.append(str(eta))
    print("general accuracy: "+str(accuracy[10]))
    resultByTest.append(str(accuracy[10]))
    print("accuracy 0: "+str(accuracy[0]))
    resultByTest.append(str(accuracy[0]))
    print("accuracy 1: "+str(accuracy[1]))
    resultByTest.append(str(accuracy[1]))
    print("accuracy 2: "+str(accuracy[2]))
    resultByTest.append(str(accuracy[2]))
    print("accuracy 3: "+str(accuracy[3]))
    resultByTest.append(str(accuracy[3]))
    print("accuracy 4: "+str(accuracy[4]))
    resultByTest.append(str(accuracy[4]))
    print("accuracy 5: "+str(accuracy[5]))
    resultByTest.append(str(accuracy[5]))
    print("accuracy 6: "+str(accuracy[6]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 7: "+str(accuracy[7]))
    resultByTest.append(str(accuracy[7]))
    print("accuracy 8: "+str(accuracy[8]))
    resultByTest.append(str(accuracy[8]))
    print("accuracy 9: "+str(accuracy[9]))
    resultByTest.append(str(accuracy[9]))
    resultByTest.append(str(endTime-startTime))
    if (hits[10]/total[10]) > best_accuracy:
        best_accuracy = hits[10]/total[10]
        best_eta = eta
    totalResult.append(resultByTest)
    #aumentando
    temp_counter = temp_counter + 1
    
write_file(totalResult, "4Changing_eta")