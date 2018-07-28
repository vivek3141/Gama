import gym
from dynamicLayers import oneLayer, twoLayers, threeLayers
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import random
import numpy as np
import cv2

print('imports done')

env = gym.make('SpaceInvaders-v0')
env.reset()

generations = 5
populationSize = 7


def checkCorrectModel(num_layers, modelParam):
    if (num_layers == 1):
        nets = oneLayer(modelParam)
    if (num_layers == 2):
        nets = twoLayers(modelParam)
    if (num_layers == 3):
        nets = threeLayers(modelParam)

    return nets


def calcScore(model):
    action = random.randrange(0, 5)
    observation, reward, done, info = env.step(action)
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, (105, 80))
    score = 0
    while True:
        action = np.argmax(model.predict(observation.reshape(-1, len(observation), 1))[0])
        observation, reward, done, info = env.step()
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (105, 80))

        score = score + reward
        if done:
            break


def fitness(modelList, modelScore, parameterList):
    max1 = np.argmax(modelScore, axis=0)
    a = [0 if i == max1 else i for i in modelScore]
    max2 = np.argmax(a, axis=0)
    ind1 = modelScore.index(max1)
    ind2 = modelScore.index(max2)
    ind3 = random.randrange(0, len(modelScore))
    while ind3 != ind1 and ind3 != ind2:
        ind3 = random.randrange(0, len(modelScore))
    return [modelList[ind1], modelList[ind2], modelList[ind3]], [parameterList[ind1], parameterList[ind2],
                                                                 parameterList[ind3]]


def mutate(models, param):
    #   neurons
    parameters = {'1': [64, 128, 256],
                  # Layers
                  '2': [1, 2, 3],
                  # Activations
                  '3': ['relu'],
                  # Optimizer
                  '4': ['rmsprop', 'adam']
                  }
    optionKeys = ['neurons', 'layers', 'activations', 'optimizers']
    mutateModelNum = random.randrange(0, len(models))
    mutateParamNum = random.randrange(1, 5)
    option = random.randrange(len(param[str(mutateParamNum)]))
    param[mutateModelNum][optionKeys[mutateParamNum]] = parameters[str(mutateParamNum)][option]

    models[mutateModelNum] = checkCorrectModel(param[mutateModelNum]['layers'][0], param[mutateModelNum])

    return models, param


def breed(p1, p2):
    ret = {}
    l = [p1[i] if random.randrange(0, 2) == 0 else p2[i] for i in p1.keys()]
    for m, k in enumerate(p1.keys()):
        ret[k] = l[m]
    return ret


def train(pop, gen, X, Y):
    trained_models = []
    for i in pop:
        model = tflearn.DNN(i, tensorboard_dir='C:/Users/rohit/Desktop/log')
        model.fit({'input': X}, {'target': Y}, n_epoch=1,

                  snapshot_step=100, show_metric=True,
                  run_id='spaceInvaders model {} out of {}, generation {}'.format(i, len(pop), gen))

        model.save('spaceInvaders_model_{}_out_of_{}_generation_{}.model'.format(i, len(pop), gen))
        trained_models.append(model)

    return trained_models


def network():
    #   neurons
    parameters = {'1': [64, 128, 256],
                  # Layers
                  '2': [1, 2, 3],
                  # Activations
                  '3': ['relu'],
                  # Optimizer
                  '4': ['rmsprop', 'adam']
                  }

    def getRandom(param):
        networkPara = {'neurons': [],
                       'layers': [],
                       'activations': [],
                       'optimizers': []}

        val1 = random.randrange(len(param[str(1)]))
        val2 = random.randrange(len(param[str(2)]))
        val3 = random.randrange(len(param[str(3)]))
        val4 = random.randrange(len(param[str(4)]))

        networkPara['neurons'].append(param['1'][val1])
        networkPara['layers'].append(param['2'][val2])
        networkPara['activations'].append(param['3'][val3])
        networkPara['optimizers'].append(param['4'][val4])

        return networkPara

    modelParam = getRandom(parameters)

    nets = checkCorrectModel(modelParam['layers'][0], modelParam)

    return nets, modelParam


def createPopulation(populationSize):
    pop = []
    popParam = []
    for i in range(populationSize):
        nets, Params = network()
        pop.append(nets)
        popParam.append(Params)
    return pop, popParam


def createNewGen(models, parameters):
    scoreList = []
    for model in models:
        scoreList.append(calcScore(model))
    fitModels, fitParameters = fitness(models, scoreList, parameters)

    parentNum = len(fitModels)
    for i in range(len(models) - len(fitModels)):
        parNum1 = random.randrange(0, parentNum)
        parNum2 = random.randrange(0, parentNum)
        bredParameters, num = breed(fitParameters[parNum1], fitParameters[parNum2])

        nets = checkCorrectModel(num, bredParameters)

        fitModels.append(bredModel)
        bredParameters.append(bredParameters)

    newGenModels, newGenParameters = mutate(fitModels, bredParameters)

    return newGenModels, newGenParameters


population, parameters = createPopulation(populationSize)
print(population, parameters)


def createVals(data):
    X = np.array([i[0] for i in data]).reshape(-1, 105, 80, 1)
    y = [i[1] for i in data]

    return X, y


for i in range(generations):
    name = '/media/vivnp/UBUNTU 18_0/Data_Gen{}.npy'.format(i + 1)
    data = np.load(name)
    X, y = createVals(data)
    trainedModels = train(population, i, X, y)
    population, parameters = createNewGen(trainedModels, parameters)
print("done")