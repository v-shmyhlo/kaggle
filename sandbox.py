from pprint import pprint as print

import torch
import torch.nn as nn

import optim

m = nn.Linear(1, 1)
opt = torch.optim.Adam(m.parameters())
opt = optim.LA(opt, 0.5, 5)
opt = optim.EWA(opt, 0.9, 10)

print(opt.state_dict())

opt.train()
m.train()
for _ in range(10):
    m(torch.ones(10, 1)).mean().backward()
    opt.step()

# print('>' * 100)
with open('s1.txt', 'w') as f:
    print(opt.state_dict(), stream=f)

state_dict = opt.state_dict()
opt = torch.optim.Adam(m.parameters())
opt = optim.LA(opt, 0.5, 5)
opt = optim.EWA(opt, 0.9, 10)
opt.load_state_dict(state_dict)

# print('>' * 100)
with open('s2.txt', 'w') as f:
    print(opt.state_dict(), stream=f)
