import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        print(f"[BUFFER] Added experience. Size: {len(self.buffer)}")
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        print(f"[BUFFER] Sampled batch of size {batch_size}")
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)