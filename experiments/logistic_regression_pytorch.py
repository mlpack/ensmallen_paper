import torch
import torch.nn as nn
from torch.optim import LBFGS
import time
import numpy as np
import sys

if len(sys.argv) != 5:
  print("args: <trainFile> <trainLabelsFile> <testFile> <testLabelsFile>")
  exit(1)

trainFile = sys.argv[1]
trainLabelsFile = sys.argv[2]
testFile = sys.argv[3]
testLabelsFile = sys.argv[4]

trainData = torch.tensor(np.genfromtxt(trainFile, delimiter=',',
    dtype=np.float64))
trainLabels = torch.tensor(np.genfromtxt(trainLabelsFile, delimiter=',',
    dtype=np.long))
testData = torch.tensor(np.genfromtxt(testFile, delimiter=',',
    dtype=np.float64))
testLabels = torch.tensor(np.genfromtxt(testLabelsFile, delimiter=',',
    dtype=np.long))

# Logistic regression model
model = nn.Linear(np.shape(trainData)[1], 2).double()

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=10)

def closure():
    optimizer.zero_grad()
    output = model(trainData)
    loss = criterion(output, trainLabels)
    loss.backward()
    return loss

start = time.perf_counter()
for i in range(10):
    optimizer.step(closure)
end = time.perf_counter()

print(f"Training took {end - start:0.6f} seconds.")

train_out = model(trainData)
_, predicted = torch.max(train_out.data, 1)
correct = (predicted == trainLabels).sum()
print("Train accuracy: ", (correct / np.shape(trainLabels)[0]))

test_out = model(testData)
_, predicted = torch.max(test_out.data, 1)
correct = (predicted == testLabels).sum()
print("Test accuracy: ", (correct / np.shape(testLabels)[0]))
