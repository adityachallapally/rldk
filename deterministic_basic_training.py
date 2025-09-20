
import sys
import random
import numpy as np
import torch

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

steps = 200
for step in range(steps):
    loss = 1.0 - (step / steps) + np.random.normal(0, 0.01)
    reward = step / steps + np.random.normal(0, 0.02)
    kl = np.random.exponential(0.1)
    
    print(f"step={step}, loss={loss:.6f}, reward_mean={reward:.6f}, kl={kl:.6f}")
