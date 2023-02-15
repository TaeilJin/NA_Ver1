import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from . import thops
from . import modules
from . import utils
from .models_HierGlow_priors import Glow
from .models_Cond_GMVAE import CondGMVAE

class HierGlow_GMVAE(nn.Module):

    def __init__(self, x_channels, cond_channels, hparams):
        super().__init__()
        self.hparams = hparams
        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        #
        self.graph = Glow(x_channels,cond_channels,hparams)
        self.graph_cond = CondGMVAE(x_channels,cond_channels,hparams.Gumbel.num_classes,hparams)

    def forward(self, x=None, cond=None, ee_cond=None, z=None,
                gumbel_temp=None,eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, cond, ee_cond,gumbel_temp)
        else:
            return self.reverse_flow(z, cond, ee_cond,gumbel_temp,eps_std)

    def normal_flow(self, x, cond, ee_cond, gumbel_temp):
        
        # condition gmvae
        total_loss, recon_loss, gaussian_loss,cate_loss, predict_label, means, var = self.graph_cond(cond=cond,label_prob=None, gumbel_temp=gumbel_temp, hard_gumbel= self.graph_cond.hard_gumbel)
        
        # flow model loss
        nBatch, nFeature, nTimesteps = x.shape
        var = var + 1e-8
        means = means.squeeze(0).reshape(nBatch,nTimesteps,-1).permute(0,2,1)
        var = var.squeeze(0).reshape(nBatch,nTimesteps,-1).permute(0,2,1)
        z, nll = self.graph(x = x,cond = cond, ee_cond=ee_cond, means = means, var = var)

        # total loss
        nll = total_loss + self.graph.loss_generative(nll)

        return z, nll, recon_loss, gaussian_loss,cate_loss, predict_label, means, var

    def reverse_flow(self, z, cond, ee_cond, gumbel_temp, eps_std):
        with torch.no_grad():

            # condition gmvae
            _,_,_,_,_,y_mu,y_var = self.graph_cond(cond=cond,label_prob=None, gumbel_temp=gumbel_temp, hard_gumbel= self.graph_cond.hard_gumbel)
            #y_mu, y_var = out_net['y_mean'], out_net['y_var']
            nBatch, nFeature, nTimesteps = cond.shape
            y_var = y_var + 1e-8
            means = y_mu.squeeze(0).reshape(nBatch,nTimesteps,-1).permute(0,2,1)
            var = y_var.squeeze(0).reshape(nBatch,nTimesteps,-1).permute(0,2,1)
            
            x = self.graph(z= z, cond = cond, ee_cond = ee_cond, eps_std=eps_std,means=means,var=var,reverse=True)
        return x
