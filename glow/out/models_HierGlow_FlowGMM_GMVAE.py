import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from . import thops
from . import modules
from . import utils
from .models_HierGlow_GMM import HierGlow_GMM
from .models_Cond_GMVAE import CondGMVAE

class HierGlow_flowGMM_GMVAE(nn.Module):

    def __init__(self, x_channels, cond_channels, hparams):
        super().__init__()
        self.hparams = hparams
        # condition encoder
        self.graph_cond = CondGMVAE(x_channels,cond_channels,hparams.Gumbel.num_classes,hparams)
        # flow model, fixed random means
        self.graph = HierGlow_GMM(x_channels,cond_channels,hparams)
        # train condition encoder or not
        self.b_train_cond = hparams.Train.train_cond_enc
        self.hard_gumbel = hparams.Gumbel.hard_gumbel
        

    def forward(self, x=None, cond=None, ee_cond=None, z=None,
                gumbel_temp=None,eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, cond, ee_cond,gumbel_temp)
        else:
            return self.reverse_flow(z, cond, ee_cond,eps_std)

    def normal_flow(self, x, cond, ee_cond, gumbel_temp):

        # prob Gaussian from condition
        if self.b_train_cond == False:
            with torch.no_grad():
                # data reshape 
                nBatch, nFeatures, nTimesteps = cond.shape
                cond = cond.permute(0,2,1)
                cond_enc = cond.reshape(-1,nFeatures).clone().detach()
                # gumbel_temp (0:categorical 1:uniform)
                out_net = self.graph_cond.network(cond_enc,gumbel_temp,self.hard_gumbel)
                prob = out_net['prob_cat']

        # gen z (mean of timesteps)
        z, nll = self.graph(x=x,cond=cond,ee_cond=ee_cond)
        # likelihood (mean of timesteps)
        mixture_ll = self.graph.loss_multiple_gaussian(z,prob)
        # total loss
        nll = -(nll +mixture_ll) 
        # mean batch size
        nll = self.graph.loss_generative(nll)

        return z, nll

    def reverse_flow(self, z, cond, ee_cond, eps_std):
        with torch.no_grad():
            #"should make sample outside model there is no reverse flow from model"
            x = self.graph(z= z, cond = cond, ee_cond = ee_cond, eps_std=eps_std,reverse=True)
        return x
