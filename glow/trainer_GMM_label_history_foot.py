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

class Trainer_Label_HISTORY_Foot(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 graph_cond,
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

    def calc_means(self, means_type="random", num_means=2, shape=66, r=1, graph_cond=None, train_loader=None, device=None):
        D = np.prod(shape)
        means = torch.zeros((num_means, D)).to(device)
        variance = torch.ones((num_means, D)).to(device)
        #num_batches = torch.zeros((num_means, 1)).to(device)

        if means_type == "random":
            for i in range(num_means):
                means[i] = r * torch.randn(D)  
        else:
            ''' use cGMVAE to compute means '''
            with torch.no_grad():
                self.graph_cond.eval()
                progress = tqdm(self.data_loader)
                n_batches =0
                for i_batch, batch in enumerate(progress):
                    # get batch data
                    for k in batch:
                        batch[k] = batch[k].to(self.data_device)
                    # get descriptor data
                    descriptor = batch["descriptor"]
                
                    nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                    des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()

                    out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)

                    # predicted
                    _, predicted_labels = torch.max(out_net['logits'], dim=1)
                    y_mu = out_net['y_mean']
                    y_var = out_net['y_var']
                    for c in range(num_means):
                        idc_list = (predicted_labels == c).nonzero(as_tuple=False)

                        if(idc_list.shape[0] > 0):
                            y_mu_c = y_mu[idc_list[:,0],:]
                            y_var_c = y_var[idc_list[:,0],:]

                            y_mu_c = thops.mean(y_mu_c,dim=[0])
                            y_var_c = thops.mean(y_var_c,dim=[0])

                            means[c] = means[c]+ y_mu_c
                            variance[c] = variance[c] + y_var_c
                            #num_batches[c] = num_batches[c] + idc_list.shape[0]    
                    n_batches = n_batches +1
                
                means = means / n_batches
                variance = variance / n_batches
                
                # for i in range(num_means):
                #     means[i] = means[i] / num_batches[i]
                #     variance[i] = variance[i] / num_batches[i]
        return means, variance
                

    def train(self):

        self.global_step = self.loaded_step
        # initial mean
        means,variance = self.calc_means(self.calc_prior,num_means = self.graph.gaussian_size, shape = self.graph.x_channels, r=1,graph_cond =self.graph_cond,train_loader=self.data_loader,device=self.data_device)
        
        if self.calc_prior == "mu":
            self.graph.means = means
        elif self.calc_prior == "mu_var":
            self.graph.means = means
            self.graph.vars = variance
            #np.savez_compressed(f'means.npz', clips = means.clone().cpu().numpy())
            #np.savez_compressed(f'variance.npz', clips = variance.clone().cpu().numpy())
        
        #self.graph.means = torch.from_numpy(np.load('means.npz')['clips'].astype(np.float32)).to(self.data_device)
        #self.graph.vars = torch.from_numpy(np.load('variance.npz')['clips'].astype(np.float32)).to(self.data_device)

        self.graph.loss_fn_GMM = SSLGaussMixture(self.graph.means, self.graph.vars, self.data_device)
        
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
                foot = batch["foot"]

                # condition encoder -> prior decoding
                with torch.no_grad():
                    self.graph_cond.eval()
                    nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                    des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                    
                    out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)
                    prob = out_net['prob_cat']
                    prob_ll = prob.clone().detach()
                    
                    prob_label = prob.reshape(nBatch_cond,nTimesteps_cond,-1).permute(0,2,1).clone().detach()

                    # for accuracy test "load 확인용"
                    #_, nFeatures_label = prob.shape
                    #label_cond = label.reshape(-1,nFeatures_label).clone().detach()
                    #accuracy, nmi = self.graph_cond.accuracy_test(predicted_labels.cpu().numpy(),label_cond[:,0].cpu().numpy())
                    
                # init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               descriptor[:self.batch_size // len(self.devices), ...] if descriptor is not None else None,
                               ee_cond[:self.batch_size // len(self.devices), ...] if ee_cond is not None else None,
                               label = prob_label,
                               foot = foot
                               )
                    # re-init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()
                
                #print("n_params: " + str(self.count_parameters(self.graph)))
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                    #self.graph_cond = torch.nn.parallel.DataParallel(self.graph_cond, self.devices, self.devices[0])
                    
                # forward phase
                z, nll = self.graph(x=x, cond=descriptor, ee_cond = ee_cond, label = prob_label, foot=foot)
                
                # multiple gaussian loss
                if hasattr(self.graph, "module"):
                    mult_gauss_ll = self.graph.module.loss_multiple_gaussian(z,prob_ll)
                else:
                    mult_gauss_ll = self.graph.loss_multiple_gaussian(z,prob_ll)
                
                nll = - (nll + mult_gauss_ll)

                if hasattr(self.graph, "module"):
                    loss_generative = self.graph.module.loss_generative(nll)
                else:
                    loss_generative =self.graph.loss_generative(nll)
                    

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
                            x_val=val_batch["x"]
                            cond_val = val_batch["cond"]
                            ee_cond_val =val_batch["ee_cond"]
                            des_val = val_batch["descriptor"]
                            foot_val = val_batch["foot"]
                            # calc cond_encoder
                            nBatch_cond, nFeatures_cond, nTimesteps_cond = des_val.shape
                            des_enc = des_val.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                            out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)
                            prob = out_net['prob_cat']

                            
                            # calc flow gmm loss
                            # init LSTM hidden
                            if hasattr(self.graph, "module"):
                                self.graph.module.init_lstm_hidden()
                            else:
                                self.graph.init_lstm_hidden()
                            label = val_batch["label"]
                            
                            prob_label = prob.reshape(nBatch_cond,nTimesteps_cond,-1).permute(0,2,1).clone().detach()
                            z_val, nll_val = self.graph(x=x_val, cond=des_val, ee_cond=ee_cond_val, label=prob_label, foot= foot_val)

                            prob_ll = prob.clone().detach()

                            # multiple gaussian loss
                            if hasattr(self.graph, "module"):
                                mult_gauss_ll = self.graph.module.loss_multiple_gaussian(z_val,prob_ll)
                            else:
                                mult_gauss_ll = self.graph.loss_multiple_gaussian(z_val,prob_ll)
                            
                            nll_val = - (nll_val + mult_gauss_ll)

                            # total loss
                            if hasattr(self.graph, "module"):
                                loss_val = loss_val + self.graph.module.loss_generative(nll_val)
                            else:
                                loss_val = loss_val + self.graph.loss_generative(nll_val)
                            
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
                         means=self.graph.means,
                         variance=self.graph.vars,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                
                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0: 
                    if self.cond_model =="enc_rot":
                        self.generator.generate_ROT_sample_withRef_cond(self.graph,gumbel_temp=self.fixed_temp_value,step=self.global_step)
                    elif self.cond_model =="enc":
                        self.generator.generate_sample_withRef_History_foot(self.graph,self.graph_cond, eps_std=1.0, step=self.global_step)



                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
