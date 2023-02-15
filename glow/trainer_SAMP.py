import re
import os
from numpy import var
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob
from .config import JsonConfig
from . import thops
from .generator import Generator
from .distributions import SSLGaussMixture

class Trainer_SAMP(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 data, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints

        # model relative
        self.graph = graph
        self.optim = optim

        self.min_gumbel = hparams.Gumbel.fixed_temp_value
        self.hard_gumbel = hparams.Gumbel.hard_gumbel
        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device

        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=1,
                                      shuffle=True,
                                      drop_last=True)
                                      
        self.n_epoches = (hparams.Train.num_batches+len(self.data_loader)-1)
        self.n_epoches = self.n_epoches // len(self.data_loader)
        self.global_step = 0
        
        self.generator = Generator(data, data_device, log_dir, hparams)

        self.calc_prior = hparams.Train.calc

        self.C1 = 52
        self.C2 = 121
        self.L = 60
        self.n_batches_train = len(self.data_loader)
        self.scheduled_sampling = True
        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                      batch_size=self.batch_size,
                                      num_workers=8,
                                      shuffle=False,
                                      drop_last=True)
            
        self.data = data
        
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        self.cond_model = hparams.Train.condmodel
        
        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap
            
    def count_parameters(self, model):
         return sum(p.numel() for p in model.parameters() if p.requires_grad)    

    def train(self):

        self.global_step = self.loaded_step
        
        # begin to train
        for epoch in range(self.n_epoches):
            print(f"epoch:{epoch} / {self.n_epoches}")
            print('Training epoch {}'.format(epoch))
            if self.scheduled_sampling:
                if epoch <= self.C1:
                    P = 1
                elif self.C1 < epoch <= self.C2:
                    P = 1 - (epoch - self.C1) / float(self.C2 - self.C1)
                else:
                    P = 0
            Bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(1 - P, dtype=torch.float))
            print('p value = {}'.format(P))
            self.writer.add_scalar('scheduled_sampling/P', P, epoch)

            progress = tqdm(self.data_loader)
            total_recon_loss =0
            total_kld_loss = 0
            total_loss = 0
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()
                
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                                                             
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                #self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                    
                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                
                prev_state = batch["prev_state"] # (Batch, nTimesteps, Feats)   
                env = batch["env"] # (Batch, nTimesteps, Feats) 
                y_state = batch["y_state"] # (Batch, nTimesteps, Feats) 

                nBatch, nFeats, nTimestep = prev_state.shape
                
                for i in range(self.L):
                    self.optim.zero_grad()
                    p = y_state[:,:,i]
                    prev = prev_state[:,:,i]
                    env_p = env[:,:,i]

                    # schedule sampling
                    if i != 0 and Bernoulli.sample().int() == 1:
                        prev = p_hat
                        prev[:, :] = torch.cat((prev[:, :66],
                                                prev_state[:, 66:, i]),
                                                dim=-1)
                    
                    p_hat, mu , logvar = self.graph(p,prev,env_p)

                    recon_loss = F.mse_loss(p_hat, p, reduction='sum') / float(nBatch)
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(nBatch)
                    loss = recon_loss + 0.1 * kld
                    
                    total_recon_loss += recon_loss.item()
                    total_kld_loss += kld.item()
                    total_loss += loss.item()

                    self.graph.zero_grad()
                    self.optim.zero_grad()
                    
                    loss.backward()
                    self.optim.step()

                
                                
                #if self.global_step % self.scalar_log_gaps == 0:
                    
                   
                    
                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                #self.optim.step()
                
                if self.global_step % self.validation_log_gaps == 0:
                    # set to eval state
                    self.graph.eval()
                                        
                    # Validation forward phase
                    loss_val = 0
                    recon_loss_val =0
                    kld_loss_val =0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            self.graph.eval()

                            val_prev_state = val_batch["prev_state"] # (Batch, nTimesteps, Feats)   
                            val_env = val_batch["env"] # (Batch, nTimesteps, Feats) 
                            val_y_state = val_batch["y_state"] # (Batch, nTimesteps, Feats) 

                            nBatch, nTimestep, nFeats = val_prev_state.shape
                            for i in range(self.L):
                                self.optim.zero_grad()
                                p = val_y_state[:,:,i]
                                prev = val_prev_state[:,:,i]
                                env_p = val_env[:,:,i]

                                # schedule sampling
                                if i != 0 and Bernoulli.sample().int() == 1:
                                    prev = p_hat
                                    prev[:, :] = torch.cat((prev[:, :66],
                                                            val_prev_state[:, 66:,i]),
                                                            dim=-1)
                                
                                p_hat, mu , logvar = self.graph(p,prev,env_p)

                                recon_loss = F.mse_loss(p_hat, p, reduction='sum') / float(nBatch)
                                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(nBatch)
                                loss = recon_loss + 0.1 * kld
                                
                                recon_loss_val += recon_loss.item()
                                kld_loss_val += kld.item()
                                loss_val += loss.item()

                               
                            n_batches = n_batches + 1  

                    recon_loss_val /= float(n_batches * self.L)
                    kld_loss_val /= float(n_batches * self.L)
                    loss_val /= float(n_batches * self.L)

                    self.writer.add_scalar("val_loss/val_loss", loss_val, self.global_step)
                    self.writer.add_scalar("val_loss/recon_loss_val", recon_loss_val, self.global_step)
                    self.writer.add_scalar("val_loss/kld_loss_val", kld_loss_val, self.global_step)
                    
                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         means=None,
                         variance=None,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                
                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0: 
                    self.generator.generate_sample_withRef_SAMP(self.graph,eps_std=1.0,step=self.global_step)
                    
                    


                # global step
                self.global_step += 1
            
            total_recon_loss /= float(self.n_batches_train * self.L)
            total_kld_loss /= float(self.n_batches_train * self.L)
            total_loss /= float(self.n_batches_train * self.L)
            #autoregressive_count /= float(self.n_batches_train * self.L)

            self.writer.add_scalar('reconstruction/train', total_recon_loss, epoch)
            self.writer.add_scalar('kld/train', total_kld_loss, epoch)
            self.writer.add_scalar('total/train', total_loss, epoch)
            #self.writer.add_scalar('scheduled_sampling/autoregressive_count_train', autoregressive_count, epoch)
            print('====> Total_train_loss: {:.4f}, recon_loss: {:.4f}, kld_loss: {:.4f}'.format(total_loss,
                                                                                                total_recon_loss,
                                                                                                total_kld_loss))
                                                                                                
            print(
                f'Loss: {total_loss:.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
