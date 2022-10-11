"""
   crown.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.hid1 = nn.Linear(2,hid)
        self.hid2 = nn.Linear(hid,hid)
        self.output = nn.Linear(hid,2)
    def forward(self, input):
        # apply network and return output
        hid1_store = self.hid1(input)
        self.hid1_sum = torch.tanh(hid1_store)
        hid2_stuff = self.hid2(self.hid1_sum)
        self.hid2_sum = torch.tanh(hid2_stuff)
        output  = torch.sigmoid(self.hid2_sum)
        return output

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        self.hid3 = None
        return 0*input[:,0]

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        return 0*input[:,0]
