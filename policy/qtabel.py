import numpy as np
from .base import Policy


import cv2

class QTable(Policy):
    def __init__(self, state_space, action_space, epsilon=0):
        self.q_table = np.random.uniform(-0.1, 0.1, size=(*state_space, action_space))
        self.action_space = np.arange(action_space)

        self.epsilon = epsilon

    def update(self, state, action, update_value):
        self.q_table[state][action] = update_value

    def _policy(self, state):
        action, value = self.epsilon_greedy(state)
        return action, value

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            q_table = cv2.GaussianBlur(self.q_table, (3, 3), 0)
            action = np.random.choice(self.action_space)
            value = q_table[state][action]
        else:
            action, value = self.max_policy(state)
        return action, value

    def max_policy(self, state):
        q_table = cv2.GaussianBlur(self.q_table, (3, 3), 0)
        #state = tuple(state)
        action = q_table[state].argmax()
        value = q_table[state].max()
        return action, value

if __name__ == "__main__":
    q = QTable(20, 3)
    print(q.q_table)
    print(q(3))