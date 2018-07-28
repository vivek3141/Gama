import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



def oneLayer(modelParam):
    network = input_data(shape=[None, 105, 80, 1], name='input')
    network = conv_2d(network, modelParam['neurons'][0], 1, activation=modelParam['activations'][0], regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 6, activation='softmax')
    network = dropout(network, 0.8)
    network = regression(network, optimizer= modelParam['optimizers'][0], learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    return network

def twoLayers(modelParam):
    network = input_data(shape=[None, 105, 80, 1], name='input')
    network = conv_2d(network,  modelParam['neurons'][0], 1, activation=modelParam['activations'][0], regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network,  modelParam['neurons'][0], activation='softmax')
    network = dropout(network, 0.8)
    network = fully_connected(network, 6, activation='softmax')
    network = dropout(network, 0.8)
    network = regression(network, optimizer= modelParam['optimizers'][0], learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    return network
def threeLayers(modelParam):
    network = input_data(shape=[None, 105, 80, 1], name='input')
    network = conv_2d(network,  modelParam['neurons'][0], 1, activation=modelParam['activations'][0], regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network,   modelParam['neurons'][0], activation='softmax')
    network = dropout(network, 0.8)
    network = fully_connected(network,  modelParam['neurons'][0], activation='softmax')
    network = dropout(network, 0.8)
    network = fully_connected(network, 6, activation='softmax')
    network = dropout(network, 0.8)
    network = regression(network, optimizer= modelParam['optimizers'][0], learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    return network