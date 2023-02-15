from tkinter import Y
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import glow.Experiment_utils as ex_utils
from  glow.Dlow_models import module_pred_z

class modelDlow(nn.Module):
    def __init__(self, x_channels, cond_channels, num_classes,hparams):
                 
        super().__init__()
        self.hparams = hparams
        self.input_size = cond_channels
        self.num_classes = num_classes
        self.gaussian_size = x_channels

        # GMVAE loss
        """
        # model generation parameters
        self.nh = nh = specs.get('nh_mlp', [300, 200]) hidden layer
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128) hidden layer
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru') 
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.fix_first = fix_first = specs.get('fix_first', False)

        cfg.lambda_kl
        cfg.lambda_j
        cfg.lambda_recon
        nk
        cfg.d_scale
        parser.add_argument('--cfg', default='h36m_nsamp10')


        nz: 128
        t_his: 25
        t_pred: 100
        lambda_v: 1000
        beta: 0.1
        vae_specs:
        rnn_type: gru
        x_birnn: false
        e_birnn: false
        use_drnn_mlp: true
        
        nk: 10
        dlow_batch_size: 64
        d_scale: 100
        lambda_j: 25
        lambda_recon: 2.0
        dlow_lr: 1.e-4
        dlow_specs:
        model_name: NFDiag
        rnn_type: gru
        nh_mlp: [1024, 512]
        x_birnn: false
        num_dlow_data_sample: 5000
        """
        self.nh_mlp = hparams.Dlow.nh_mlp
        self.nh_rnn = hparams.Dlow.nh_rnn    
        self.rnn_type = hparams.Dlow.rnn_type
        self.x_birnn = hparams.Dlow.x_birnn 
        self.fix_first = hparams.Dlow.fix_first

        self.lambda_j = hparams.Dlow.lambda_j
        self.lambda_recon = hparams.Dlow.lambda_recon
        self.lambda_kl = hparams.Dlow.lambda_kl
        self.nSample = hparams.Dlow.nSample
        self.d_scale = hparams.Dlow.d_scale
     
        # gumbel
        self.init_temp = hparams.Gumbel.init_temp
        self.hard_gumbel = hparams.Gumbel.hard_gumbel
        self.gumbel_temp = self.init_temp
        
        # 
        self.network = module_pred_z.GRU_MappingFunc(cond_channels,x_channels,self.nSample,self.nh_rnn,self.nh_mlp)
    
    def init_lstm_hidden(self):
        self.network.init_hidden()
        
    def joint_loss_world(self,Y_g,cond_gt,scaler):
        Y_g = Y_g.permute(1,0,3,2).clone().detach() #(nBatch,nSample,nFpose,ntimesteps) -> (nSample,nBatch,,ntimesteps,nFpose) 
        #(nBatch,nTimesteps,nFeats)
        cond_gt = cond_gt.permute(0,2,1)
    
       
        loss =0.0

        return loss

    def joint_loss(self,Y_g):
        loss = 0.0
        nBatch = Y_g.shape[0]
        self.d_scale = 1.0
        Y_g = Y_g.reshape(nBatch,self.nSample,-1)
        for Y in Y_g:
            dist = F.pdist(Y, 2) ** 2
            loss += (-dist /self.d_scale ).exp().mean()
        loss /= Y_g.shape[0]
        return loss

    def recon_loss(self,Y_g, Y):
        #(Batch,nSample, timesteps, Fpose) 
        #(Batch,timesteps,Fpose)
        diff = Y_g - Y.unsqueeze(1)
        #(Batch,nSample, timesteps, Fpose)
        diff = diff.reshape(Y_g.shape[0],self.nSample,-1)
        #(Batch,nSample,Fmotion)
        dist = diff.pow(2).sum(dim=-1)#.sum(dim=0) #(Batch,nSample)
        loss_recon = dist.min(dim=1)[0].mean() #(Batch,1)
        return loss_recon

    def loss_function(self, Y_g, Y, a, b, cond_gt=None,scaler=None):
        #(Batch,nSample, timesteps, Fpose) 
        # KL divergence loss
        KLD = self.network.get_kl(a, b) / a.shape[0]
        # joint loss
        if scaler ==None:
            JL = self.joint_loss(Y_g) if self.lambda_j > 0 else 0.0
        else:
            JL = self.joint_loss_world(Y_g,cond_gt,scaler)
        # reconstruction loss
        RECON = self.recon_loss(Y_g, Y) if self.lambda_recon > 0 else 0.0
        loss_r = KLD * self.lambda_kl + JL * self.lambda_j + RECON * self.lambda_recon
        return loss_r, np.array([loss_r.item(), KLD.item(), 0 if JL == 0 else JL.item(), 0 if RECON == 0 else RECON.item()])


    def forward(self,cond=None):
        # data reshape 
        cond_input = cond.clone().detach().permute(0,2,1) # (B,T,Fcond)
        
        Z, a, b = self.network(cond_input) # (B,T,Fcond) -> (B, nk,Fpose)

        return Z, a,b


   