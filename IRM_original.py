# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:48:16 2022

@author: Ark

original paper IRM 
"""

import torch
from torch.autograd import grad

torch.manual_seed(10)

def compute_penalty(losses,dummy_w):
    g1 = grad(losses[0::2].mean(), dummy_w, create_graph = True )[0]
    g2 = grad(losses[1::2].mean(), dummy_w, create_graph = True )[0]
    return (g1*g2).sum()

# this codes the causal relationship of x,y,z
def example_1 (n =10000 , d =2 , env =1):
    x = torch.randn(n,d)*env
    y = x + torch.randn(n,d)*env
    z = y + torch.randn(n,d)
    return torch.cat((x,z) , 1) , y.sum (1, keepdim = True )

environments = [example_1(env =0.1),example_1(env =1.0)]

  
# IRM version
phi = torch.nn.Parameter(torch.ones(4 , 1))
dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))
opt = torch.optim.SGD ([phi], lr=1e-3)
mse = torch.nn.MSELoss(reduction ="none")


opt.zero_grad()
print('\n---------IRM from paper---------')
print('torch.manual_seed(10)')
for iteration in range(5000):
    error = 0
    penalty = 0
    for x_e , y_e in environments :
        p = torch.randperm(len( x_e ))
        y_model = x_e @ phi * dummy_w
        error_e = mse(y_model[p], y_e[p])
        penalty += compute_penalty(error_e , dummy_w)
        error += error_e.mean()
        
    opt.zero_grad()
    (1e-5 * error + penalty).backward()
    opt.step()
    
    if iteration % 1000 == 0:
        print('weights of x0,x1,z0,z1',phi)

        
   