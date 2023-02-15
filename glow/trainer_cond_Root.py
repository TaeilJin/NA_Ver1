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
from .models_HierGlow_GMM_ENV import HierGlow_GMM_ENV
from . import thops
from .generator import Generator
from .distributions import SSLGaussMixture

class Trainer_Foot_Estimator(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 graph_cond,
                 graph_pose,
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

        self.graph_cond = graph_cond
        self.graph_pose = graph_pose
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
                
                foot = batch["foot"]
                descriptor = batch["descriptor"]

                # condition encoder -> prior decoding
                with torch.no_grad():
                    self.graph_cond.eval()
                    nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                    des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                    
                    out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)
                    prob = out_net['prob_cat']
                    
                    # for accuracy test "load 확인용"
                    #_, nFeatures_label = prob.shape
                    #label_cond = label.reshape(-1,nFeatures_label).clone().detach()
                    #accuracy, nmi = self.graph_cond.accuracy_test(predicted_labels.cpu().numpy(),label_cond[:,0].cpu().numpy())
                    
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                    #self.graph_cond = torch.nn.parallel.DataParallel(self.graph_cond, self.devices, self.devices[0])
                    
                # forward phase
                out_foot = self.graph(cond=descriptor, phi = prob.clone().detach())
                foot_loss = self.graph.gen_foot_loss(out_foot = out_foot, gt_foot=foot)
                                   
                loss = torch.mean(foot_loss)

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
                    acc_val =0
                    nmi_val =0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            self.graph_cond.eval()
                            self.graph.eval()

                            # get validation data
                            des_val = val_batch["descriptor"]
                            foot_val = val_batch["foot"]
                            
                            # calc cond_encoder
                            nBatch_cond, nFeatures_cond, nTimesteps_cond = des_val.shape
                            des_enc = des_val.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                            out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)
                            prob = out_net['prob_cat']
                            
                            # foot estimator
                            # forward phase
                            out_foot_val = self.graph(cond=des_val, phi = prob.clone().detach())
                            foot_loss_val = self.graph.gen_foot_loss(out_foot = out_foot_val, gt_foot=foot_val)
                                   
                            nll_val = torch.mean(foot_loss_val)


                            # total loss
                            if hasattr(self.graph, "module"):
                                loss_val = loss_val + nll_val
                            else:
                                loss_val = loss_val + nll_val
                            
                            # accuracy test ("확인용")  
                            # _, predicted_labels = torch.max(out_net['logits'], dim=1)
                            # _, _, nFeatures_label = label.shape
                            # label_cond = label.reshape(-1,nFeatures_label).clone().detach()
                            #accuracy, nmi = self.graph_cond.accuracy_test(predicted_labels.cpu().numpy(),label_cond[:,0].cpu().numpy())
                            #acc_val = acc_val + accuracy
                            #nmi_val = nmi_val + nmi

                            n_batches = n_batches + 1        
                    
                    loss_val = loss_val/n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                
                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0: 
                    if self.cond_model =="enc_rot":
                        self.generator.generate_ROT_sample_withRef_cond(self.graph,gumbel_temp=self.fixed_temp_value,step=self.global_step)
                    elif self.cond_model =="enc":
                        self.generator.generate_sample_withRef_foot_estimator(self.graph_pose,self.graph_cond,self.graph, eps_std=1.0, step=self.global_step)
                    


                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
