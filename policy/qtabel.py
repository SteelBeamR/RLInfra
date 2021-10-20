import numpy as np
from base import Policy

class QTable(Policy):
    def __init__(self, state_space, action_space):
        self.q_table = np.random.uniform(-2, 0, size=(state_space, action_space))

    def update(self, state, update_value):
        self.q_table[state] = update_value

    def _policy(self, state):
        action = self.q_table[state].argmax()
        value = self.q_table[state].max()
        return action, value

if __name__ == "__main__":
    q = QTable(20, 3)
    print(q.q_table)
    print(q(3))