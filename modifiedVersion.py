import os
os.chdir("C:\\Users\\jeong\\Documents\\Python Scripts\\neural-networks-and-deep-learning\\python3Version\\DeepLearningPython35")

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

import network

mini_batch_size = 10

t_data, validation_data, t_data = load_data_wrapper()
out_data = list(t_data)
topology = [784,300,30,10]
net = network.Network(topology)
epochs=60
eta=0.1
net.SGD(training_data=out_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=eta, test_data=validation_data)


#%% 
t_data, validation_data, t_data = load_data_wrapper()
final_test_data = list(t_data)
accuracy = np.zeros((11,1))
result = net.evaluate(t_data)
#for x in final_test_data:
#    result = net.evaluate()
    

#print("final result: ")
#print("mini_batch_size: "+mini_batch_size)
#print("topology: "+topology)
#print("epochs: "+epochs)
#print("eta: "+eta)
#print("general accuracy: "+accuracy[10])
#print("accuracy 0: "+accuracy[0])
#print("accuracy 1: "+accuracy[1])
#print("accuracy 2: "+accuracy[2])
#print("accuracy 3: "+accuracy[3])
#print("accuracy 4: "+accuracy[4])
#print("accuracy 5: "+accuracy[5])
#print("accuracy 6: "+accuracy[6])
#print("accuracy 7: "+accuracy[7])
#print("accuracy 8: "+accuracy[8])
#print("accuracy 9: "+accuracy[9])

