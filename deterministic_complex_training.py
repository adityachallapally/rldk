
import sys
import random
import numpy as np
import torch
import torch.nn as nn

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

steps = 100
for step in range(steps):
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    reward = -loss.item() + np.random.normal(0, 0.01)
    kl = np.random.exponential(0.05)
    
    print(f"step={step}, loss={loss.item():.6f}, reward_mean={reward:.6f}, kl={kl:.6f}")
