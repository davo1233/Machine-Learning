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
        self.hid1_layer = nn.Linear(2,hid)
        self.hid2_layer = nn.Linear(hid,hid)
        self.output = nn.Linear(hid,1)
    def forward(self, input):
        # apply network and return output
        hid1_store = self.hid1_layer(input)
        self.hid1 = torch.tanh(hid1_store)
        hid2_store= self.hid2_layer(self.hid1)
        self.hid2 = torch.tanh(hid2_store)       
        output_store = self.output(self.hid2)
        output = torch.sigmoid(output_store)
        return output

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.hid1_layer = nn.Linear(2,hid)
        self.hid2_layer = nn.Linear(hid,hid)
        self.hid3_layer = nn.Linear(hid,hid)
        self.output = nn.Linear(hid,1)
    def forward(self, input):
        # apply network and return output
        hid1_store = self.hid1_layer(input)
        self.hid1 = torch.tanh(hid1_store)
        hid2_store= self.hid2_layer(self.hid1)
        self.hid2 = torch.tanh(hid2_store)
        hid3_store= self.hid3_layer(self.hid2)
        self.hid3 = torch.tanh(hid3_store)   
        output_store = self.output(self.hid3)
        output = torch.sigmoid(output_store)
        return output

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.hid1_layer = nn.Linear(2,num_hid)
        self.hid2_layer = nn.Linear(num_hid + 2,num_hid)
        self.output = nn.Linear(num_hid + num_hid + 2,1)
    def forward(self, input):
        # apply network and return output
        hid1_store = self.hid1_layer(input)
        self.hid1 = torch.tanh(hid1_store)
        hid2_store = self.hid2_layer(torch.cat((self.hid1,input),1))
        self.hid2 = torch.tanh(hid2_store)       
        output_store = self.output(torch.cat((self.hid2,self.hid1,input),1))
        output1 = torch.sigmoid(output_store)
        return output1
