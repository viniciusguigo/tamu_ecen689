class RandomAgent(object):
    """ Performs random actions.
    Used to test environment action space.
    """
    def __init__(self, env):
        self.name = 'RandomAgent'
        print('\n*** AGENT: {} ***\n'.format(self.name))
        self.env = env
        self.action_space = env.action_space

    def act(self, observation, reward=None, done=None):
        # take a random action
        return self.action_space.sample()

    def close(self):
        """ Stop any thread (if any) or save additional data (if any)"""
        pass
