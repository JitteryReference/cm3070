from pettingzoo.classic import chess_v4
import time
import random
import numpy as np

env = chess_v4.env()

steps = 0
env.reset()
t = time.time()
moves = list([i for i in range(4672)])
for agent in env.agent_iter():
    observation, reward, done, info = env.last()

    if done:
    	break

    print(np.array(observation['observation']).shape)
    break
    mask = observation['action_mask']
    env.step(np.random.choice(moves, p=mask/np.sum(mask)))
    #env.render()
    steps += 1

print('steps per second:', round(steps/(time.time()-t), 2))
print('steps:', steps)
