import gym
import time

env = gym.make("MountainCar-v0")
state = env.reset()
done = False
fps = 100
while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    print(reward, new_state, done, _)
    time.sleep(1/fps)
    env.render()


class Agent:
    def __init__(self, policy=None, action=None, update=None):
        self.policy = policy
        self.action = action
        self.update = update

class policy

class Env


