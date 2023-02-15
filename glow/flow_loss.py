import numpy as np
import torch.nn as nn
import torch
from . import thops
class FlowLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior, k=256):
        super().__init__()
        self.k = k
        self.prior = prior
    
    def forward(self, z, sldj, y=None, device=None):
        # z = z.reshape((z.shape[0], ))
        n_Batch, n_Features, n_Timesteps = z.shape 
        prior_ll = torch.zeros((n_Batch,n_Timesteps),dtype=float,device=device)
        z = torch.transpose(z,1,2)
        if y is not None:
            y = torch.transpose(y,1,2)
            
        for b in range(0,n_Batch):
            z_batch = z[b,...]
            if y is not None:
                y_batch = y[b,...]
                y_batch = y_batch.squeeze_(1)
                prior_ll[b,:] = self.prior.log_prob(z_batch, y_batch)
            else:
                prior_ll[b,:] = self.prior.log_prob(z_batch)
        prior_ll = thops.sum(prior_ll, dim=[1])
        
        prior_ll = sldj + prior_ll
        prior_nll = torch.div( (-1 *prior_ll) , float(np.log(2.) * n_Timesteps) )

        # n_Batch, n_Features, n_Timesteps = z.shape 
        # prior_ll = torch.zeros((n_Batch,n_Timesteps),dtype=float,device=device)
        # for k in range(0,n_Timesteps):
        #     z_timestep = z[...,k]
        #     y_timestep = y[...,k]
        #     y_timestep = y_timestep.squeeze_(1)
        #     if y_timestep is not None:
        #         prior_ll[:,k] = self.prior.log_prob(z_timestep, y_timestep)
        #     else:
        #         prior_ll[:,k] = self.prior.log_prob(z_timestep)
        # prior_ll = thops.sum(prior_ll, dim=[1])
        # #consistency loss


        # prior_ll = sldj + prior_ll
        # prior_nll = (-prior_ll) / float(np.log(2.) * n_Timesteps)
        
        #another correction? corrected_prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:]) 
        
        return prior_nll
