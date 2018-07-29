import gym
import random
import cv2
import numpy as np

env = gym.make('SpaceInvaders-v0')
env.reset()

minScore = 100
MaxSteps = 600
minEpisodes = 250
numGenerations = 5


def processGameData(game_data, training_data):
    for data in game_data:
        if (data[1] == 0):
            output = [1, 0, 0, 0, 0, 0]
            training_data.append([data[0], output])
        elif (data[1] == 1):
            output = [0, 0, 1, 0, 0, 0]
            training_data.append([data[0], output])
        elif (data[1] == 2):
            output = [0, 0, 0, 1, 0, 0]
            training_data.append([data[0], output])
        elif (data[1] == 3):
            output = [0, 0, 0, 1, 0, 0]
            training_data.append([data[0], output])
        elif (data[1] == 4):
            output = [0, 0, 0, 0, 1, 0]
            training_data.append([data[0], output])
        elif (data[1] == 5):
            output = [0, 0, 0, 0, 0, 1]
            training_data.append([data[0], output])

    return training_data


for i in range(1, numGenerations + 1):

    trainingData = []
    scores = []

    gameData = []
    for i in range(minEpisodes):
        env.reset()
        score = 0
        gameData = []
        for _ in range(MaxSteps):

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = cv2.resize(observation, (105, 80))
            gameData.append([observation, action])
            score = score + reward

            if done:
                break

        if (score >= minScore):
            scores.append(score)
            trainingData = processGameData(gameData, trainingData)
        print(i)

    training_data_save = np.array(trainingData)
    np.save('Data_Gen{}.npy'.format(i), training_data_save)

    if (len(scores) != 0):
        print(mean(scores))