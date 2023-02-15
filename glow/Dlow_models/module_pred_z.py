import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from glow.Dlow_models.mlp import MLP

def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')


# Here we define our model as a class
class LSTM_MappingFunc(nn.Module):

    def __init__(self, input_dim,output_dim,num_data, hidden_dim, nh_mlp, num_layers=1):
        super(LSTM_MappingFunc, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_data   = num_data
        self.num_layers = num_layers
        
        # Define the LSTM layer (B,T,Fcond) -> (B,T,128)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        # Define MLP (B,1,128) -> tanh(Linear) -> (B,1,1024) -> tahnh(Linear) -> (B,1,512)
        self.mlp_hn = MLP(self.hidden_dim, nh_mlp)
        # Define OutLayer A (B,1,512) -> Linear -> (B,1,Fpose *num_data)
        self.head_A = nn.Linear(nh_mlp[-1], output_dim * num_data)
        # Define OutLayer b (B,1,512) -> Linear -> (B,1,Fpose *num_data)
        self.head_b = nn.Linear(nh_mlp[-1], output_dim * num_data)

        # do_init
        self.do_init = True

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (batch_size, num_layers, hidden_dim).
        if self.do_init:
            gru_out, self.hidden = self.gru(input)
            self.do_init = False
        else:
            gru_out, self.hidden = self.gru(input, self.hidden) #(B,T,Fcond) -> (B,T,128)
        
        # get last output (B,1,128)
        gru_out = gru_out[:,-1,:] # (B,128)

        # (B,128) ->  (B,512)
        gru_out = self.mlp_hn(gru_out)

        # Final layer A (B,512) ->  (B,66 * num_data)
        y_A = self.head_A(gru_out)
        y_b = self.head_b(gru_out)
        
        # calculate sampled z
        if z is None:
            z = torch.randn((input.shape[0], self.output_dim), device=input.device)
        Z = z.repeat_interleave(self.num_data, dim=1) # (B,66*num_data)

        Z = y_A * Z + y_b
        return Z, y_A, y_b



# Here we define our model as a class
class GRU_MappingFunc(nn.Module):

    def __init__(self, input_dim,output_dim,num_data, hidden_dim, nh_mlp, num_layers=1):
        super(GRU_MappingFunc, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_data   = num_data
        self.num_layers = num_layers
        
        # Define the LSTM layer (B,T,Fcond) -> (B,T,128)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        # Define MLP (B,1,128) -> tanh(Linear) -> (B,1,1024) -> tahnh(Linear) -> (B,1,512)
        self.mlp_hn = MLP(self.hidden_dim, nh_mlp)
        # Define OutLayer A (B,1,512) -> Linear -> (B,1,Fpose *num_data)
        self.head_A = nn.Linear(nh_mlp[-1], output_dim * num_data)
        # Define OutLayer b (B,1,512) -> Linear -> (B,1,Fpose *num_data)
        self.head_b = nn.Linear(nh_mlp[-1], output_dim * num_data)

        # do_init
        self.do_init = True

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, input, z =None):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (batch_size, num_layers, hidden_dim).
        if self.do_init:
            gru_out, self.hidden = self.gru(input)
            self.do_init = False
        else:
            gru_out, self.hidden = self.gru(input, self.hidden) #(B,T,Fcond) -> (B,T,128)
        
        # get last output (B,1,128)
        gru_out = gru_out[:,-1,:] # (B,128)

        # (B,128) ->  (B,512)
        gru_out = self.mlp_hn(gru_out)

        # Final layer A (B,512) ->  (B,66 * num_data)
        y_A = self.head_A(gru_out)
        y_b = self.head_b(gru_out)
        
        # calculate sampled z
        if z is None:
            z = torch.randn((input.shape[0], self.output_dim), device=input.device)
        Z = z.repeat_interleave(self.num_data, dim=1) # (B,66*num_data)

        Z = y_A * Z + y_b
        return Z, y_A, y_b
    
    def get_kl(self, a, b):
        var = a ** 2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return KLD