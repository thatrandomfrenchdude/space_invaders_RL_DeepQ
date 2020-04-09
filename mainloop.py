import gym
from spaceinvaders import DeepQNetwork, Agent
from RL_utils import plotLearning
import numpy as np

if __name__ == '__main__':
    print('Starting...')
    env = gym.make('SpaceInvaders-v0')
    brain = Agent(gamma=0.95, epsilon=1.0,
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)
    print('Initializing memory with random gameplay...')
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # ACTIONS: 0 noop, 1 fire, 2 right, 3 left, 4 right fire, 5 left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), action, reward, np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
    print('Memory initialization complete! Learning...')

    scores = []
    epsHistory = []
    numGames = 100
    batchSize = 32 # large batch -> better training, less performance

    for i in range(numGames):
        print('game ', i + 1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200, 30:125], axis=2)]
        score = 0
        lastAction = 0

        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = lastAction

            observation_, reward, done, infor = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200, 30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward -= 100
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), action, reward, np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
            brain.learn(batchSize)
            lastAction = action
            # env.render()
        scores.append(score)
        print('score: ', score)
        # x = [i + 1 for i in range(numGames)]
        # filename = 'test' + str(numGames) + '.png'
        # plotLearning(x, scores, epsHistory, filename)
