#!/usr/bin/env python
"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

"""

import torch
from torch.autograd import Variable


def sinkhorn_normalized(x, y, epsilon, mu, nu, n, m, niter):
    """
    Computes the Sinkhorn divergence. It consists in substracting the transport
    cost of moving the distribution itself. It ensures that the cost of transporting
    a distribution to itself is zero.
    """
    Wxy = sinkhorn_loss(x, y, epsilon, mu, nu, n, m, niter)
    Wxx = sinkhorn_loss(x, x, epsilon, mu, mu, n, n, niter)
    Wyy = sinkhorn_loss(y, y, epsilon, nu, nu, m, m, niter)
    return Wxy - 0.5 * Wxx - 0.5 * Wyy


def sinkhorn_loss(x, y, epsilon, mu, nu, n, m, niter=100, acc=1e-3, unbalanced=False, gpu=False):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    
    INPUTS:
        x : positions of diracs for the first distribution, torch.FloatTensor of size [n, d]
        y : positions of diracs for the second distribution, torch.FloatTensor of size [m, d]
        epsilon : importance of the entropic regularization
        mu : mass located at each dirac, torch.FloatTensor of size [n]
        nu : mass located at each dirac, torch.FloatTensor of size [m]
        n : total number of diracs of the first distribution
        m : total number of diracs of the second distribution
        niter : maximum number of Sinkhorn iterations
        acc : required accuracy to satisfy convergence
        unbalanced : specify if unbalanced OT needs to be solved
        gpu : specify usage of CUDA with pytorch
    
    OUTPUTs:
        cost : the cost of moving from distribution x to y
    """
    # The Sinkhorn algorithm takes as input three variables :
    C = Variable(cost_matrix(x, y), requires_grad=True)  # Wasserstein cost function

    # use GPU if asked to
    if(gpu & torch.cuda.is_available()):  
        C = C.cuda()
        mu = nu.cuda()
        nu = nu.cuda()
        

    # Parameters of the Sinkhorn algorithm.
    tau = -.8  # nesterov-like acceleration
    thresh = acc  # stopping criterion
    if(unbalanced):
        rho = 1  (.5) **2          # unbalanced transport
        lam = rho / (rho + epsilon)  # Update exponent
        

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.repeat(m,1).transpose(0,1) + v.repeat(n,1)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = torch.zeros_like(mu), torch.zeros_like(nu) , 0.
    u.requires_grad=True
    v.requires_grad=True
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        if(unbalanced):
            # accelerated unbalanced iterations
            u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
            v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        else:
            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.data.unsqueeze(1)
    y_lin = y.data.unsqueeze(0)
#    x_col = x.unsqueeze(1)
#    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c
