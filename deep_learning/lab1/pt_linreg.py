import torch
import torch.nn as nn
import torch.optim as optim

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])
optimizer = optim.SGD([a, b], lr=0.1)
for i in range(100):
    Y_ = a*X + b
    diff = (Y-Y_)
    loss = torch.mean(diff**2)
    loss.backward()
    optimizer.step()
    print("Gradient of a:{}".format(a.grad.item()))
    print("Gradient of b:{}".format(b.grad.item()))
    optimizer.zero_grad()
    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')