import re
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob
from .config import JsonConfig
from .models_HierGlow_GMVAE import Glow
from . import thops
from .generator import Generator


class Trainer_total(object):
    def __init__(self, model, optim, lrschedule, loaded_step,
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
        self.model = model
        self.optim = optim
        
        self.fixed_temp = hparams.Gumbel.fixed_temp
        self.fixed_temp_value = hparams.Gumbel.fixed_temp_value
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

        
        # self.seqlen = hparams.Data.seqlen
        # self.n_lookahead = hparams.Data.n_lookahead
        
        ##test batch
        # self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      # batch_size=self.batch_size,
                                       # num_workers=1,
                                      # shuffle=False,
                                      # drop_last=True)
        # self.test_batch = next(iter(self.test_data_loader))
        # for k in self.test_batch:
            # self.test_batch[k] = self.test_batch[k].to(self.data_device)

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
            print("epoch", epoch)
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.model.train()
                
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

                # init LSTM hidden
                if hasattr(self.model.graph, "module"):
                    self.model.graph.module.init_lstm_hidden()
                else:
                    self.model.graph.init_lstm_hidden()
                
                if self.fixed_temp == True:
                    gumbel_temp = self.fixed_temp_value
                else:
                    gumbel_temp = self.model.graph_cond.update_temperature(epoch)


                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.model(x[:self.batch_size // len(self.devices), ...],
                               cond[:self.batch_size // len(self.devices), ...] if cond is not None else None,
                               ee_cond[:self.batch_size // len(self.devices), ...] if ee_cond is not None else None,
                               gumbel_temp = gumbel_temp)
                    # re-init LSTM hidden
                    if hasattr(self.model.graph, "module"):
                        self.model.graph.module.init_lstm_hidden()
                    else:
                        self.model.graph.init_lstm_hidden()
                
                #print("n_params: " + str(self.count_parameters(self.graph)))
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.model.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.model = torch.nn.parallel.DataParallel(self.model, self.devices, self.devices[0])
                    
                # forward phase
                z, loss_generative, recon_loss, gaussian_loss,cate_loss, predict_label, means, var = self.model(x=x, cond=cond, ee_cond = ee_cond, gumbel_temp=gumbel_temp)

                #
                if self.global_step % 10 == 0:
                    nBatch_cond, nFeatures_cond, nTimesteps_cond = cond.shape
                    enc_cond = cond.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                    out_net = self.model.graph_cond.network(enc_cond,gumbel_temp,self.model.graph_cond.hard_gumbel)
                    z = out_net["gaussian"]
                    z = z.squeeze(0).reshape(nBatch_cond,nTimesteps_cond,-1).permute(0,2,1)
                    x_bar = self.model(z=z, cond=cond, ee_cond = ee_cond, eps_std=1.0, gumbel_temp=gumbel_temp,reverse=True)
                    loss = (x - x_bar).pow(2)
                    
                    recon_x_loss = thops.mean(loss.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).sum(-1),dim=[0])
                    loss_generative += recon_x_loss

                # with torch.no_grad():
                #     _, nFeatures_label, _ = label.shape
                #     label_cond = label.permute(0,2,1).reshape(-1,nFeatures_label).clone().detach()
                #     # classification loss               
                #     accuracy, nmi = self.model.graph_cond.accuracy_test(predict_label.cpu().numpy(),label_cond[:,0].cpu().numpy())

                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                    self.writer.add_scalar("loss/recon_loss", recon_loss, self.global_step)
                    self.writer.add_scalar("loss/gaussian_loss", gaussian_loss, self.global_step)
                    self.writer.add_scalar("loss/cate_loss", cate_loss, self.global_step)
                    self.writer.add_scalar("loss/recon_x_loss", recon_x_loss, self.global_step)
                    #self.writer.add_scalar("loss/nmi", nmi, self.global_step)

                loss = loss_generative

                # backward
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                
                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()
                
                if self.global_step % self.validation_log_gaps == 0:
                    # set to eval state
                    self.model.eval()
                                        
                    # Validation forward phase
                    loss_val = 0
                    n_batches = 0
                    loss_recon_val = 0
                    loss_cate_val = 0
                    loss_gaussian_val = 0
                    recon_x_val = 0
                    acc_nmi = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            self.model.graph_cond.eval()
                            cond = val_batch["cond"]
                            ee_cond =val_batch["ee_cond"]
                            # init LSTM hidden
                            if hasattr(self.model.graph, "module"):
                                self.model.graph.module.init_lstm_hidden()
                            else:
                                self.model.graph.init_lstm_hidden()
                            label = val_batch["label"]
                            z, nll_val, recon_loss, gaussian_loss,cate_loss, predict_label, means, var = self.model(x=val_batch["x"], cond=cond, ee_cond=ee_cond, gumbel_temp = gumbel_temp)
                            
                            # # classification loss
                            # # _, nFeatures_label, _ = label.shape
                            # label_cond = label.permute(0,2,1).reshape(-1,nFeatures_label).clone().detach()               
                            # accuracy, nmi = self.model.graph_cond.accuracy_test(predict_label.cpu().numpy(),label_cond[:,0].cpu().numpy())
                            
                            nBatch_cond, nFeatures_cond, nTimesteps_cond = cond.shape
                            enc_cond = cond.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                            out_net = self.model.graph_cond.network(enc_cond,gumbel_temp,self.model.graph_cond.hard_gumbel)
                            z = out_net["gaussian"]
                            z = z.squeeze(0).reshape(nBatch_cond,nTimesteps_cond,-1).permute(0,2,1)
                            x_bar = self.model(z=z, cond=cond, ee_cond = ee_cond, eps_std=1.0, gumbel_temp=gumbel_temp,reverse=True)
                            loss = (x - x_bar).pow(2)
                            
                            recon_x_loss = thops.mean(loss.permute(0,2,1).reshape(-1,66).sum(-1),dim=[0])
                                                        
                            # loss
                            loss_val = loss_val +nll_val
                            loss_recon_val = loss_recon_val + recon_loss
                            loss_cate_val = loss_cate_val + cate_loss
                            loss_gaussian_val = loss_gaussian_val + gaussian_loss
                            
                            recon_x_val = recon_x_val + recon_x_loss
                            #acc_nmi = acc_nmi + nmi
                            n_batches = n_batches + 1   

                    
                    loss_val = loss_val/n_batches
                    loss_recon_val = loss_recon_val/n_batches
                    loss_cate_val = loss_cate_val/n_batches
                    loss_gaussian_val = loss_gaussian_val/n_batches
                    recon_x_val = recon_x_val/n_batches
                    #acc_nmi = acc_nmi/n_batches
                    

                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                    self.writer.add_scalar("val_loss/loss_recon_val", loss_recon_val, self.global_step)
                    self.writer.add_scalar("val_loss/loss_cate_val", loss_cate_val, self.global_step)
                    self.writer.add_scalar("val_loss/loss_gaussian_val", loss_gaussian_val, self.global_step)
                    self.writer.add_scalar("val_loss/recon_x_val", recon_x_val, self.global_step)
                    #self.writer.add_scalar("val_loss/acc_nmi", acc_nmi, self.global_step)

                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.model,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                         
                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0:  
                    self.generator.generate_sample_withRef(self.model.graph,self.model.graph_cond, eps_std=1.0, step=self.global_step)

                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
