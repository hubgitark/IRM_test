# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:43:15 2022

@author: Ark

from 

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

original is binary label
v0: 
    moving some functions outside the loop
    WIP update final result record to pandas
    add basic training curve visualization
    
v1:
    added tricolor builder env
        show pred vs true on image binary

v2:
    adding full label builder for original env - currently on this

v3:
    WIP adding new env builder for even - red, odd - green
    

"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
from torchvision import datasets
from torch import nn, optim, autograd

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# a few global controller
isLabelFlip = True #for training
isColorFlip = True #for training
isIRM = True

isLabelFlip_OOD = False
isColorFlip_OOD = False



# Define loss function helpers
# nll = negative log likelihood, eq. cross-ent
def mean_nll(logits, y):
    # <----specific, change here for binary vs multi-class
    #return nn.functional.binary_cross_entropy_with_logits(logits, y)
    return nn.CrossEntropyLoss()(logits.float(), y)

def mean_accuracy(logits, y):
    # <----specific, change here for binary vs multi-class
    #preds = (logits > 0.).float()
    preds = logits.argmax(dim=1).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("     ".join(str_values))

def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1

# Build environments, even - red, odd – green exp, tri-color, full label
def make_environment(images, labels, e):

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]  #torch.Size([10000, 14, 14])

    # Assign a binary label based on the digit; flip label with probability 0.25
    if isLabelFlip:
        labels = labels + torch_bernoulli(0.1, len(labels)) #<---PARAM
        labels = torch.clamp(labels,min=0.0,max=9.0)
        
    # Assign a color based on the label; flip the color with probability e
    if not isColorFlip: e = 0
    colors = torch_xor((labels%2==0).float(), torch_bernoulli(e, len(labels)))
    
    # Apply the color to the image by zeroing out the other color channel
    # note rgb channel =  [red, green, blue]
    images = torch.stack([images, images, torch.zeros(images.shape)], dim=3)
    images[torch.tensor(range(len(images))), :, :,(1-colors).long()] *= 0

    labels = labels.type(torch.LongTensor)
    return {
            'images': (images.float() / 255.).cuda(),
            'labels': labels.cuda()
            }   


# Build environments, all red, tri-color, full label
def make_environment_OOD(images, labels, e):
    labels_ori = labels.clone()    
    
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]  #original binary: torch.Size([10000, 14, 14])

    # Assign a binary label based on the digit; flip label with probability 0.25
    # this is the type I OOD part:
    labels = (labels < 0).float()
    if isLabelFlip_OOD: 
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        
    # Assign a color based on the label; flip the color with probability e
    if not isColorFlip_OOD: e = 0 #<--- PARAM
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    
    # Apply the color to the image by zeroing out the other color channel
    # note rgb channel =  [red, green, blue]
    images = torch.stack([images, images, torch.zeros(images.shape)], dim=3)
    images[torch.tensor(range(len(images))), :, :,(1-colors).long()] *= 0
    
    labels = labels_ori.clone()
    return {
            'images': (images.float() / 255.).cuda(),
            'labels': labels.cuda()
            }   



### -------------------------Data Prep and Model Training------------------------ ###
parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))
    
'''
# a tensorflow version of mnist for debugging
import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#train_images, train_labels = train_images[:40000], train_labels[:40000] #if full set, memory overflow
#test_images, test_labels = test_images[:10000], test_labels[:10000] #if full set, memory overflow

mnist_train = (torch.from_numpy(train_images[:40000]), torch.from_numpy(train_labels[:40000]))
mnist_val = (torch.from_numpy(train_images[40000:50000]), torch.from_numpy(train_labels[40000:50000]))
#mnist_OOD = (torch.from_numpy(train_images[50000:]), torch.from_numpy(train_labels[50000:]))
mnist_OOD = (torch.from_numpy(test_images), torch.from_numpy(test_labels))

'''

# Load MNIST, make train/val splits, and shuffle train set examples
mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:40000], mnist.targets[:40000])
mnist_val = (mnist.data[40000:50000], mnist.targets[40000:50000])
mnist_OOD = (mnist.data[50000:], mnist.targets[50000:])


# OOD testing
test_env = make_environment_OOD(mnist_OOD[0], mnist_OOD[1], 0.9)
test_images = test_env['images']#.cpu().numpy()
test_labels = test_env['labels'].cpu().numpy().T


final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)
    
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]

    writer = SummaryWriter()
    # Define and instantiate the model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(3 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 10)
            # commented out for random initial weighting
            #for lin in [lin1, lin2, lin3]:
                #nn.init.xavier_uniform_(lin.weight)
                #nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 3, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 3 * 14 * 14)
            out = self._main(out)
            return out

    mlp = MLP().cuda()


    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    lossLst = []
    train_accLst = []
    test_accLst = []
    #training loop
    for step in range(flags.steps):
    
        for env in envs:

            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        # from original code, a loss function modifier, can comment out no noticible impact
        #loss += flags.l2_regularizer_weight * weight_norm

        # IRM portion 
        # penalty up from 1 to 10000 when steps more than 100 by original
        if isIRM:
            penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculating acc on env[2] aka test
        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )
        
        lossLst.append(loss)
        train_accLst.append(train_acc)
        test_accLst.append(test_acc)
        
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    

    
### -------------------------Visualization of Last Result------------------------ ###

plt.plot(train_accLst, label='train_acc')
plt.plot(test_accLst, label = 'test_acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend(loc='lower right')
plt.show()
plt.clf()
#loss is not very meaningful for IRM since dynamic penalty_multiplier


#------result review (full label)------\
test_pred = mlp(test_images)
test_pred = test_pred.argmax(dim=1).detach().cpu().numpy()
test_pred_show = test_pred.copy()
test_pred_boo = pd.Series((test_pred == test_labels))


#print a few pred vs test, first 25 wrongs
test_pred_boo_wrong_index = test_pred_boo[test_pred_boo==False].index
test_images_wrong = test_images.cpu().numpy()[test_pred != test_labels]

test_pred_wrong = pd.Series(test_pred_show[test_pred_boo_wrong_index]).astype(str)
test_pred_wrong_actual = pd.Series(test_labels[test_pred_boo_wrong_index]).astype(str)
test_pred_wrong_vs_actual = test_pred_wrong.str.cat(test_pred_wrong_actual,sep=' pred vs actual ')


plt.figure(figsize=(10,8))
for i in range(min(20,len(test_images_wrong))):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # test_images_wrong[i] is shape (2,14,14) # original
    # need to change to (14,14,3)
    plt.imshow(test_images_wrong[i])
    plt.xlabel(test_pred_wrong_vs_actual[i])
plt.show()
plt.clf()


#print a few pred vs test, first 25 corrects
test_pred_boo_correct_index = test_pred_boo[test_pred_boo==True].index
test_images_correct = test_images.cpu().numpy()[test_pred == test_labels]

test_pred_correct = pd.Series(test_pred_show[test_pred_boo_correct_index]).astype(str)
test_pred_correct_actual = pd.Series(test_labels[test_pred_boo_correct_index]).astype(str)
test_pred_correct_vs_actual = test_pred_correct.str.cat(test_pred_correct_actual,sep=' pred vs actual ')

plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_correct[i])
    plt.xlabel(test_pred_correct_vs_actual[i])
plt.show()
plt.clf()


print('test_OOD:')
print('test_OOD_pred_wrong',str(len(test_pred_wrong))+'/'+str(len(test_pred)),
      'test_OOD_pred_correct',str(len(test_pred_correct))+'/'+str(len(test_pred)),)




    
