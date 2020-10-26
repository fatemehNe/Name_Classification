from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.optim as optim
from lstm import RNN

rnn = torch.load("LSTM64.pt")

A = list(rnn.lstm.named_parameters())

name = ["IT.txt", "FT.txt", "ZT.txt" , "OT.txt"]
nameL = "linearT.txt"

########## lstm weights ######

# w weights
x = A[0]
g = torch.split(x[1],n_hidden)
for m in range(4):
    fT = open(name[m], "a")
    h = torch.transpose(g[m], 0, 1)
    for i in range(n_letters):
        for j in range(n_hidden):
            fT.write(h[i][j].item()+"\n") 
    fT.close()

# # U weights
x = A[1]
g = torch.split(x[1],n_hidden)
for m in range(4):
    fT = open(name[m], "a")
    h = torch.transpose(g[m], 0, 1)
    for i in range(n_hidden):
        for j in range(n_hidden):
            fT.write(h[i][j].item()+"\n") 
    fT.close()

# # bias 
x = A[2][1] + A[3][1]
g = torch.split(x,n_hidden)
for m in range(4):
    fT = open(name[m], "a")
    for j in range(n_hidden):
        fT.write(g[m][j].item() +"\n") 
    fT.close()

# linear weights
B = list(rnn.hidden2Cat.named_parameters())

x = B[0]
fT = open(nameL, "a")
h = torch.transpose(x[1], 0, 1)
for i in range(n_hidden):
    for j in range(n_categories):
        fT.write(h[i][j].item()+"\n") 
fT.close()

x = B[1]
fT = open(nameL, "a")
for i in range(n_categories):
    fT.write(x[1][i].item()+"\n") 
fT.close()

