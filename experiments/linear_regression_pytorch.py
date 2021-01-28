import torch
import torch.nn as nn
from torch.optim import LBFGS
import time
import numpy as np
import sys

if len(sys.argv) != 3:
  print("args: <dim> <points>")
  exit(1)

dim = int(sys.argv[1])
points = int(sys.argv[2])

x = torch.Tensor(np.random.rand(points, dim)).double()
y = torch.Tensor(np.random.rand(points, 1)).double()

# Logistic regression model
model = nn.Linear(np.shape(x)[1], 1).double()

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=10)

def closure():
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    return loss

start = time.perf_counter()
for i in range(10):
    optimizer.step(closure)
end = time.perf_counter()

print((end - start))
