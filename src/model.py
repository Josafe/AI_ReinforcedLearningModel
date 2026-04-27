import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, debug=False):
        super().__init__()
        self.debug = debug

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        if self.debug:
            print(f"[MODEL] Input: {x}")

        x = F.relu(self.fc1(x))
        if self.debug:
            print(f"[MODEL] After fc1: {x}")

        x = F.relu(self.fc2(x))
        if self.debug:
            print(f"[MODEL] After fc2: {x}")

        output = self.out(x)

        if self.debug:
            print(f"[MODEL] Output (Q-values): {output}")

        return output
    
