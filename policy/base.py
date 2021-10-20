class Policy:
    def __call__(self, state):
        return self._policy(state)

    def update(self, state, update_value):
        pass

    def _policy(self, state):
        pass

