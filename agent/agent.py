class Agent:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.state = self.env.reset()

    def policy(self, state):
        action, value = self.policy(state)
        return action, value

    def action(self, action):
        new_state, reward, done, _ = env.step(action)
        

