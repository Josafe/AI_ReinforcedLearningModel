import torch
import torch.optim as optim
import random
import numpy as np
from src.model import DQN

print("DQN loaded from:", DQN.__module__)
print("DQN init args:", DQN.__init__.__code__.co_varnames)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_dim, action_dim, debug=False).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.epsilon = 1.0

        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            print("f[AGENT] Random action:{action}")
            return action
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.model(state)
            action = q_values.argmax().item()

        print(f"[AGENT] Policy action: {action}")
        return action
    
    def train_step(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        #Q(s,a)
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s,a)
        next_q_values = self.target_model(next_states).max(1)[0]

        expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.functional.mse_loss(q_value, expected_q.detach())

        print (f"[TRAIN] Loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("[AGENT] Target network updated")