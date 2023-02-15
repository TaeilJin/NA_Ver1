import torch
import torch.nn as nn
from glow.Networks_EnvEncoder import *
from glow.LossFunctions import *
from glow.Metrics import *
from glow.models_SAMP import FootContactNet
import numpy as np
class FOOT_estimator(nn.Module):
    def __init__(self, x_channels, cond_channels, num_classes,hparams):
                 
        super().__init__()
        self.hparams = hparams
        self.input_size = cond_channels
        self.num_classes = num_classes
        self.gaussian_size = x_channels

        # GMVAE loss
        """
        ## Loss function parameters
        parser.add_argument('--w_gauss', default=1, type=float,
                            help='weight of gaussian loss (default: 1)')
        parser.add_argument('--w_categ', default=1, type=float,
                            help='weight of categorical loss (default: 1)')
        parser.add_argument('--w_rec', default=1, type=float,
                            help='weight of reconstruction loss (default: 1)')
        parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                            default='bce', help='desired reconstruction loss function (default: bce)')
        """
        self.w_cat = hparams.Gumbel.w_categ
        self.w_gauss = hparams.Gumbel.w_gauss    
        self.w_rec = hparams.Gumbel.w_rec
        self.rec_type = hparams.Gumbel.rec_type 
        self.w_foot = hparams.Gumbel.w_foot

        # gumbel
        self.init_temp = hparams.Gumbel.init_temp
        self.decay_temp = hparams.Gumbel.decay_temp
        self.hard_gumbel = hparams.Gumbel.hard_gumbel
        self.min_temp = hparams.Gumbel.min_temp
        self.decay_temp_rate = hparams.Gumbel.decay_temp_rate
        self.verbose = hparams.Gumbel.verbose
        self.gumbel_temp = self.init_temp
        
        self.foot_estimator = FootContactNet(rng = np.random.RandomState(23456))
        self.criterion = nn.BCELoss()
        self.m_sig = nn.Sigmoid()
        self.m_softmax = nn.Softmax(dim=-1)
    
    def gen_foot_loss(self,out_foot, gt_foot):
        _, fFeatures, _ = gt_foot.shape
        foot = gt_foot.permute(0,2,1).reshape(-1,fFeatures).clone().detach()
        loss_foot = self.criterion(self.m_sig(out_foot[:,:2]), foot[:,:2]) + self.criterion(self.m_sig(out_foot[:,2:]), foot[:,2:])
        return loss_foot

    def forward(self,cond=None,phi=None):
        # scene feature reshape 
        nBatch, nFeatures, nTimesteps = cond.shape
        cond = cond.permute(0,2,1)
        cond = cond.reshape(-1,nFeatures).clone().detach()
        
        # foot
        out_foot = self.foot_estimator(cond[:,:693],cond[:,-2640:],phi)
        
        
        return out_foot
