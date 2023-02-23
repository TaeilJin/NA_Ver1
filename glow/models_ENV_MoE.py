import torch
import torch.nn as nn
from glow.Networks_EnvEncoder import *
from glow.LossFunctions import *
from glow.Metrics import *
from glow.models_SAMP import PoseCVAE
import torch.nn.functional as F
import numpy as np

class cVAE_Gating(nn.Module):
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
        self.batch_size = hparams.Train.batch_size
        self.cVAE = PoseCVAE(rng = np.random.RandomState(23456))
        self.criterion = nn.BCELoss()
        self.m_sig = nn.Sigmoid()
        self.m_softmax = nn.Softmax(dim=-1)
    
    def addEndEffectorElement(self,z, ee_cond):
        upper_ee_idx =[
                15,16,17, 
                27,28,29,
                39,40,41,
                (17)*3+0,(17)*3+1,(17)*3+2,
                (21)*3+0,(21)*3+1,(21)*3+2
                ]
        # 순서가 Head LH, RH 라는 것을 기억하자
        for ii, index in enumerate(upper_ee_idx):
            z[:,index] = ee_cond[:,ii] 
        return z
        
    def gen_cVAE_loss(self,x_hat, x_gt, mu, logvar):
       
        recon_loss = F.mse_loss(x_hat, x_gt, reduction='sum') / float(self.batch_size)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(self.batch_size)
        loss = recon_loss + 0.1 * kld

        return loss

    def forward(self,x,cond, ee_cond):
        # pose reshape
        nBatch, nFeatures, nTimesteps = x.shape
        x = x.permute(0,2,1)
        x = x.reshape(-1,nFeatures).clone().detach()
        
        # scene feature reshape 
        nBatch, nFeatures, nTimesteps = cond.shape
        cond = cond.permute(0,2,1)
        cond = cond.reshape(-1,nFeatures).clone().detach()
        
        ee_cond = ee_cond.permute(0,2,1)
        ee_cond = ee_cond.reshape(nBatch*nTimesteps,-1).clone().detach()

        masked_ee = torch.zeros((nBatch*nTimesteps,66)).to(ee_cond.device)
        ee_cond = self.addEndEffectorElement(masked_ee,ee_cond)
        # foot
        x_hat, mu,logvar = self.cVAE(x,cond,ee_cond)
        
        loss = self.gen_cVAE_loss(x_hat, x, mu, logvar)
        return x_hat, loss
