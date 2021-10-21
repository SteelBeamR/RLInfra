import gym
import time

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation



from agent.agent import Agent
from policy.qtabel import QTable

size = [200, 200]
bin_size = [1.8 / size[0], 0.14 / size[1]]
def state_bin(state):
    state = [int((state[0]+1.2) / bin_size[0]),  int((state[1]+0.07) / bin_size[1])]
    return tuple(state)

policy = QTable((size[0], size[1]), 3, epsilon=0.1)

env = gym.make("MountainCar-v0")
agent = Agent(policy, env, lr=0.2, discount=0.99)

plt.ion()

fig = plt.figure(tight_layout=True)
show = plt.imshow(agent.policy.q_table[..., 0])


done = False
FPS = 200
max_height = 0
max_velo = 0
for i in range(500):
    state = env.reset()
    state = state_bin(state)
    cnt = 0 
    while not done:
        new_state, value, reward, done, action, _ = agent(state)
        new_state = state_bin(new_state)
        if state[0]>max_height:
            max_height = state[0]
            height_reward = 500
        else:
            height_reward = 0
        
        if abs(state[1] - size[1] //2) > max_velo:
            max_velo = abs(state[1] - size[1] //2)
            velo_reward = 500
        else:
            velo_reward = 0
        
        
        out_reward = 0
        done_reward = 100 * (reward + 1)
        
        #out_reward, done_reward = 0, 0

        motion = (new_state[0] - max_height) + (abs(new_state[1] - size[1] //2) - max_velo)
        reward = motion + height_reward + velo_reward + out_reward + done_reward
        reward = reward / 20
        print(i, cnt, state, new_state, reward, max_height, max_velo)

        agent.update(state, new_state, action, value, reward)
        state = new_state 
        cnt += 1
        if cnt==190:
            break
        env.render()

        #time.sleep(1/FPS)
        if cnt %50 == 0:
            img = agent.policy.q_table
            ran = img.max() - img.min()
            img = (img - img.min()/ ran)
            img = cv2.GaussianBlur(img, (13, 13), 0)
            plt.imshow(np.log(img+1))
            plt.pause(1/FPS)
    plt.clf()
