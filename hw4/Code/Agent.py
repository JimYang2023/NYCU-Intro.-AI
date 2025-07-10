import random
import numpy as np

class Agent:
    def __init__(self,k,epsilon,constant=None):
        self.size = k
        self.epsilon = epsilon
        self.constant = constant
        self.reset()

    def reset(self):        
        self.q_values = np.zeros(self.size)
        self.action_counts = np.zeros(self.size)
        
    def select_action(self):
        p = random.uniform(0,1)
        if p < self.epsilon:
            return random.randint(0, self.size - 1)
        return np.argmax(self.q_values)

    def update_q(self, action, reward):
        if self.constant == None:        
            self.action_counts[action] += 1
            p = 1 / float(self.action_counts[action])
            self.q_values[action] += p * (reward - self.q_values[action])
        else:
            self.q_values[action] += self.constant * (reward - self.q_values[action])
