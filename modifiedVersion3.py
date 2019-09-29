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

result = []

mini_batch_size = 10
epochs = 60
learning_rate = 0.03
regularization_factor = 0.1
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



#------------------------------------------------------------------------------
mini_batch_size = 10
epochs = 60
learning_rate = 0.03
regularization_factor = 0.1
topology = [
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 4, 4),
                  filter_shape=(60, 20, 2, 2),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)]

net = Network(topology, mini_batch_size)
result.append(net.SGD(training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, lmbda=regularization_factor))

dump_file(result, "result_pickle")




