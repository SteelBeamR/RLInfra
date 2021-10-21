import numpy as np

class Agent:
    def __init__(self, policy, env, lr=0.8, discount=0.99):
        self.policy = policy
        self.env = env
        self.state = self.env.reset()

        self.lr = lr
        self.discount = discount

    def policy(self, state):
        action, value = self.policy(state)

        return action, value

    def action(self, action):
        new_state, reward, done, _ = self.env.step(action)
        return new_state, reward, done, _

    def __call__(self, state):
        action, value = self.policy(state)
        new_state, reward, done, _ = self.action(action)
        return new_state, value, reward, done, action, _

    def update(self, state, new_state, action, value, reward):
        new_action, new_value = self.policy(new_state)
        new_value = (1 - self.lr)*value + self.lr * (reward + self.discount * new_value)
        self.policy.update(state, action, new_value)
