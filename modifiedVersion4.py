def testTheano():
    from theano import function, config, shared, sandbox
    import theano.tensor as T
    import numpy
    import time
    print("Testing Theano library...")
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
# Perform check:
#testTheano()

def write_file(listt, name):
    f= open(name+".csv","w+")
    for i in listt:
        for j in i:
            f.write(str(j)+"; ")
        f.write("\n")

def dump_file(listt, name):
    f= open(name+".pickle","wb+")
    pickle.dump(listt, f)


# ----------------------
# - network3.py example:
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

# read data:
training_data, validation_data, test_data = network3.load_data_shared()


# chapter 6 -  rectified linear units and some l2 regularization (lmbda=0.1) => even better accuracy
from network3 import ReLU
import pickle
import numpy as np
#%%
result = []
filter_size=1
best_result=0

#mini_batch_size = 10
#epochs = 90
#learning_rate = 0.03
#regularization_factor = 0.1
#topology = [
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                  filter_shape=(20, 1, 1, 1),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    ConvPoolLayer(image_shape=(mini_batch_size, 20, 14, 14),
#                  filter_shape=(40, 20, 1, 1),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#    SoftmaxLayer(n_in=100, n_out=10)]
#

#net = Network(topology, mini_batch_size)
#result.append(net.SGD(training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, lmbda=regularization_factor))
#dump_file(result, "result_pickle"+str(filter_size))


result_mini_batch = []
best_number_mini_batch = 0
mini_batch_size = 1
epochs = 5
learning_rate = 0.03
regularization_factor = 0.1
for j in range(0,10):
    mini_batch_size = mini_batch_size + j
    result = []
    topology = [
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)]
    
    net = Network(topology, mini_batch_size)
    result.append(net.SGD(training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, lmbda=regularization_factor))
    dump_file(result, "result_pickle_mini_batch"+str(j))
    result_mini_batch.append(0)
    for d in range(0, len(result[0])):
         if result[0][d]>result_mini_batch[j-1]:
             result_mini_batch[j-1] = result[0][d]

temp_batch = 1
for temp in range(0, len(result_mini_batch)):
    temp_batch = temp_batch + temp
    if result_mini_batch[temp]>result_mini_batch[best_number_mini_batch]:
        best_number_mini_batch = temp_batch
         

result_learning = []
best_number_learning = 0
mini_batch_size = best_number_mini_batch + 1
epochs = 5
learning_rate = 0.03
regularization_factor = 0.1
for j in range(1,15):
    learning_rate = j*0.3
    result = []
    topology = [
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)]
    
    net = Network(topology, mini_batch_size)
    result.append(net.SGD(training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, lmbda=regularization_factor))
    dump_file(result, "result_pickle_learning"+str(j))
    result_learning.append(0)
    for d in range(0, len(result[0])):
         if result[0][d]>result_learning[j-1]:
             result_learning[j-1] = result[0][d]

for temp in range(0, len(result_learning)):
    if result_learning[temp]>result_learning[best_number_learning]:
        best_number_learning = temp




result_regularization = []
best_number_regularization = 0
learning_rate = 0.03*(best_number_learning+1)
regularization_factor = 0.1
for j in range(1,15):
    regularization_factor = j*0.1
    result = []
    topology = [
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)]
    
    net = Network(topology, mini_batch_size)
    result.append(net.SGD(training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, lmbda=regularization_factor))
    dump_file(result, "result_pickle_regularization"+str(j))
    result_regularization.append(0)
    for d in range(0, len(result[0])):
         if result[0][d]>result_regularization[j-1]:
             result_regularization[j-1] = result[0][d]

for temp in range(0, len(result_learning)):
    if result_regularization[temp]>result_regularization[best_number_regularization]:
        best_number_regularization = temp




best_epoch_number = 0
best_epoch_result = 0
epochs = 160
result = []
topology = [
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)]

net = Network(topology, mini_batch_size)
result.append(net.SGD(training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, lmbda=regularization_factor))
dump_file(result, "result_pickle_epoch"+str(j))
result_regularization.append(0)
for d in range(0, len(result[0])):
     if result[0][d]>best_epoch_result:
         best_epoch_result = result[0][d]
         best_epoch_number = d+1



    




