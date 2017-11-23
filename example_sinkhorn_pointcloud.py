#!/usr/bin/env python
"""
example_sinkhorn_pointcloud.py

Minimal example for point cloud OT

"""

import numpy as np
import matplotlib.pyplot as plt
import torch

import sinkhorn_pointcloud as spc


# Inspired from Numerical tours : Point cloud OT
from numpy import random

n = 200
N = [n,n] # Number of points per cloud

# Dimension of the cloud : 2
x = random.rand(2,N[0])-.5
theta = 2*np.pi*random.rand(1,N[1])
r = .8 + .2*random.rand(1,N[1])
y = np.vstack((np.cos(theta)*r,np.sin(theta)*r))
plotp = lambda x,col: plt.scatter(x[0,:], x[1,:], s=50, edgecolors="k", c=col, linewidths=1)

# Plot the marginals
plt.figure(figsize=(6,6))
plotp(x, 'b')
plotp(y, 'r')
# plt.axis("off")
plt.xlim(np.min(y[0,:])-.1,np.max(y[0,:])+.1)
plt.ylim(np.min(y[1,:])-.1,np.max(y[1,:])+.1)
plt.title("Input marginals")

# Sinkhorn parameters
epsilon = 0.01
niter = 100

# Wrap with torch tensors
X = torch.FloatTensor(x.T)
Y = torch.FloatTensor(y.T)

l1 = spc.sinkhorn_loss(X,Y,epsilon,n,niter)
l2 = spc.sinkhorn_normalized(X,Y,epsilon,n,niter)

print("Sinkhorn loss : ", l1.data[0])
print("Sinkhorn loss (normalized) : ", l2.data[0])

plt.show()
