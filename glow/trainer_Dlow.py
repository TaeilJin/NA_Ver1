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
from .models_Dlow import modelDlow
from . import thops
from .generator import Generator
from .distributions import SSLGaussMixture

class Trainer_Dlow(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 graph_gen,graph_cond,
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
        self.checkpoints_gap = hparams.Dlow.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints

        # model relative
        self.graph = graph
        self.optim = optim

        self.graph_cond = graph_cond
        self.graph_gen = graph_gen
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
                                      
        self.n_epoches = (hparams.Dlow.num_batches+len(self.data_loader)-1)
        self.n_epoches = self.n_epoches // len(self.data_loader)
        self.global_step = 0
        
        self.generator = Generator(data, data_device, log_dir, hparams)

        self.calc_prior = hparams.Train.calc

        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                      batch_size=self.batch_size,
                                      num_workers=8,
                                      shuffle=False,
                                      drop_last=True)
            
        self.data = data
        self.data_dir = hparams.Dir.data_dir
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Dlow.validation_log_gap
        self.plot_gaps = hparams.Dlow.plot_gap
            
    def count_parameters(self, model):
         return sum(p.numel() for p in model.parameters() if p.requires_grad)    

    def train(self):

        self.global_step = self.loaded_step
        
        # begin to train
        for epoch in range(self.n_epoches):
            print(f"epoch:{epoch} / {self.n_epoches}")
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()
                
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                                                             
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                    
                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                
                x = batch["x"]          
                cond = batch["cond"]
                ee_cond = batch["ee_cond"]
                label = batch["label"]
                descriptor = batch["descriptor"]

                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                
                self.graph.init_lstm_hidden()
                self.graph_gen.init_lstm_hidden()
                # control -> generate Z 
                gen_z_value, a, b = self.graph(cond) # (B,Fpose*num_data)
                nBatch, nFeats_nData = gen_z_value.shape
                nFpose = nFeats_nData // self.graph.nSample
                gen_z_value = gen_z_value.reshape(nBatch, self.graph.nSample,-1) #(B, numData, Fpose)

                with torch.no_grad():
                    self.graph_gen.eval()
                    self.graph_cond.eval()
                        
                    nBatch, nFdes, nTimesteps = descriptor.shape
                    nD_sampled = torch.zeros((nBatch,self.graph.nSample,nFpose, nTimesteps)).to(self.data_device)
                    for nD in range(self.graph.nSample):
                        gen_z_eps = gen_z_value[:,nD,:] #(B,Fpose)

                        # input z epsilion value
                        gen_z_eps = gen_z_eps.repeat_interleave(nTimesteps, dim=0).unsqueeze(1) #(BxTimesteps,Fpose)
                        gen_z_eps = gen_z_eps.repeat_interleave(self.graph_gen.gaussian_size,dim=1) #(BxTimesteps,classess,Fpose)
                        # probablitiy 
                        des_cond = descriptor.permute(0,2,1).reshape(-1,nFdes).clone().detach()
                        out_net = self.graph_cond.network(des_cond,0.7,self.graph_cond.hard_gumbel)
                        prob = out_net['prob_cat'] # (BxTimesteps,classess)
                        prob = prob.unsqueeze(1)
                        # means, vars
                        means_nD = self.graph_gen.means.clone().detach() #(classess,Fpose)
                        vars_nD = self.graph_gen.vars.clone().detach() 
                        means_nD = means_nD.unsqueeze(0).repeat_interleave(nBatch*nTimesteps,dim=0).to(self.data_device) #(BxTimesteps,classess,Fpose)
                        vars_nD = vars_nD.unsqueeze(0).repeat_interleave(nBatch*nTimesteps,dim=0).to(self.data_device) #(BxTimesteps,classess,Fpose)
                        # calc input
                        gen_z_eps = means_nD + gen_z_eps * vars_nD #(BxTimesteps,classess,Fpose)
                        gen_z_eps = torch.bmm(prob,gen_z_eps) #(BxTimesteps,1,classess) x (BxTimesteps,classess,Fpose) = (BxTimesteps,1,Fpose)
                        #
                        gen_z_eps = gen_z_eps.reshape(nBatch,-1,nFpose).permute(0,2,1) # (B,Fpose,Timesteps)
                        sampled = self.graph_gen(z=gen_z_eps, cond=cond, ee_cond = ee_cond, eps_std=1.0, reverse=True)
                        nD_sampled[:,nD,:,:] = sampled

                # loss
                # nD_sampled (nSample, nBatch, Fpose, timesteps)  
                # gt (nBatch, Fpose, timesteps)   
                loss_generative, loss_items = self.graph.loss_function(nD_sampled,x,a,b,cond)
               
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                    self.writer.add_scalar("loss/loss_total", loss_items[0], self.global_step)
                    self.writer.add_scalar("loss/loss_KL", loss_items[1], self.global_step)
                    self.writer.add_scalar("loss/loss_JL", loss_items[2], self.global_step)
                    self.writer.add_scalar("loss/loss_RECON", loss_items[3], self.global_step)
                               

                loss = loss_generative

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()
                
                if self.global_step % self.validation_log_gaps == 0:
                    # set to eval state
                    self.graph.eval()
                                        
                    # Validation forward phase
                    loss_val = 0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            self.graph_gen.eval()
                            self.graph_cond.eval()
                            self.graph.eval()

                            # get validation data
                            x_val=val_batch["x"]
                            cond_val = val_batch["cond"]
                            ee_cond_val =val_batch["ee_cond"]
                            des_val = val_batch["descriptor"]
                            
                            # calc flow gmm loss
                            self.graph.init_lstm_hidden()
                            
                            # control -> generate Z 
                            gen_z_value, a, b = self.graph(cond_val) # (B,Fpose*num_data)
                            nBatch, nFeats_nData = gen_z_value.shape
                            nFpose = nFeats_nData // self.graph.nSample
                            gen_z_value = gen_z_value.reshape(nBatch, self.graph.nSample,-1) #(B, numData, Fpose)

                            nBatch, nFdes, nTimesteps = des_val.shape
                            nD_sampled = torch.zeros((nBatch,self.graph.nSample,nFpose, nTimesteps)).to(self.data_device)
                            for nD in range(self.graph.nSample):
                                gen_z_eps = gen_z_value[:,nD,:] #(B,Fpose)

                                # input z epsilion value
                                gen_z_eps = gen_z_eps.repeat_interleave(nTimesteps, dim=0).unsqueeze(1) #(BxTimesteps,Fpose)
                                gen_z_eps = gen_z_eps.repeat_interleave(self.graph_gen.gaussian_size,dim=1) #(BxTimesteps,classess,Fpose)
                                # probablitiy 
                                des_cond = des_val.permute(0,2,1).reshape(-1,nFdes).clone().detach()
                                out_net = self.graph_cond.network(des_cond,0.7,self.graph_cond.hard_gumbel)
                                prob = out_net['prob_cat'] # (BxTimesteps,classess)
                                prob = prob.unsqueeze(1)
                                # means, vars
                                means_nD = self.graph_gen.means.clone().detach() #(classess,Fpose)
                                vars_nD = self.graph_gen.vars.clone().detach() 
                                means_nD = means_nD.unsqueeze(0).repeat_interleave(nBatch*nTimesteps,dim=0).to(self.data_device) #(BxTimesteps,classess,Fpose)
                                vars_nD = vars_nD.unsqueeze(0).repeat_interleave(nBatch*nTimesteps,dim=0).to(self.data_device) #(BxTimesteps,classess,Fpose)
                                # calc input
                                gen_z_eps = means_nD + gen_z_eps * vars_nD #(BxTimesteps,classess,Fpose)
                                gen_z_eps = torch.bmm(prob,gen_z_eps) #(BxTimesteps,1,classess) x (BxTimesteps,classess,Fpose) = (BxTimesteps,1,Fpose)
                                #
                                gen_z_eps = gen_z_eps.reshape(nBatch,-1,nFpose).permute(0,2,1) # (B,Fpose,Timesteps)
                                sampled = self.graph_gen(z=gen_z_eps, cond=cond_val, ee_cond = ee_cond_val, eps_std=1.0, reverse=True)
                                nD_sampled[:,nD,:,:] = sampled

                            
                            
                            # loss
                            # nD_sampled (nSample, nBatch, Fpose, timesteps)  
                            # gt (nBatch, Fpose, timesteps)   
                            loss_generative, val_loss_items = self.graph.loss_function(nD_sampled,x_val,a,b,cond_val)
                            loss_val = loss_val + loss_generative
                            n_batches = n_batches + 1        
                    
                    loss_val = loss_val/n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                    self.writer.add_scalar("loss/loss_KL", val_loss_items[1], self.global_step)
                    self.writer.add_scalar("loss/loss_JL", val_loss_items[2], self.global_step)
                    self.writer.add_scalar("loss/loss_RECON", val_loss_items[3], self.global_step)

                   
                #self.generator.generate_diverse_motion_withDlow(self.graph,self.graph_gen,self.graph_cond, eps_std=1.0, step=self.global_step)
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         means=self.graph_gen.means,
                         variance=self.graph_gen.vars,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)

                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0:   
                    self.generator.generate_diverse_motion_withDlow("Demo_lc_ld_b.txt", self.data_dir,self.graph,self.graph_gen,self.graph_cond, eps_std=1.0, step=self.global_step)

                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
