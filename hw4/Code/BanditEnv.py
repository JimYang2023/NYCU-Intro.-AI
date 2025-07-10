import numpy as np

class BanditEnv:
    def __init__(self,k,stationary=True):
        self.size = k
        self.stationary = stationary
        self.reset()

    def reset(self):
        self.actions = []
        self.rewards = []
        self.opt_actions = []
        self.means = np.random.randn(self.size)
        
    def step(self,action):
        if not self.stationary:
            self.means += np.random.normal(loc=0,scale=0.01,size=self.size)
        reward = np.random.normal(self.means[action],scale=1.0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.opt_actions.append(np.argmax(self.means))
        return reward

    def export_history(self):
        return self.actions, self.rewards
    
