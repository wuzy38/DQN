import random

class ReplayMemory():
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.position = 0
        self.buffer = []
        
    def add(self, transition_tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition_tuple
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)