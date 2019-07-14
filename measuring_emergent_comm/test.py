import numpy as np
import torch.distributions as dists
import torch

probs = torch.Tensor([0.1, 0.2, 0.3, 0.4])
m = dists.Categorical(probs)
action = m.sample()
print(action)
logprobs = m.log_prob(action)
print(logprobs)

# 0 -2.3026
# 1 -1.6094
# 2 -1.2040
# 3 -0.9163