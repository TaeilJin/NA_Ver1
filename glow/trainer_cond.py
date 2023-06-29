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
from . import thops
from .generator_cond import Generator_Cond


class Trainer_Cond(object):
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
        
        """ Model """
        self.fixed_temp = hparams.Gumbel.fixed_temp
        self.fixed_temp_value = hparams.Gumbel.fixed_temp_value
        self.cond_model = hparams.Train.condmodel

        self.default_steps = 100000
        self.N = self.default_steps // len(self.data_loader)
        graph.decay_temp_rate = (graph.init_temp - graph.min_temp) /self.N
        


        """ TEST data & visualization """

        self.generator = Generator_Cond(data, data_device, log_dir, hparams)
      

        # graph.decay_temp_rate = (graph.init_temp - graph.min_temp) /self.n_epoches
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
        self.val_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=hparams.Test.batch_size,
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
                descriptor = batch["sf"]
                
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                    
                # forward phase
                if self.fixed_temp == True:
                    gumbel_temp = self.fixed_temp_value
                else:
                    gumbel_temp = self.graph.update_temperature(epoch)
                
                total_loss, recon_loss, gaussian_loss,cate_loss, predict_label = self.graph(cond=descriptor,label_prob=None, gumbel_temp=gumbel_temp, hard_gumbel= self.graph.hard_gumbel)
                
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/total_loss", total_loss, self.global_step)
                    self.writer.add_scalar("loss/recon_loss", recon_loss, self.global_step)
                    self.writer.add_scalar("loss/guassian_loss", gaussian_loss, self.global_step)
                    self.writer.add_scalar("loss/cate_loss", cate_loss, self.global_step)
                    self.writer.add_scalar("info/gumbel_temp", gumbel_temp, self.global_step)
                   
                loss = total_loss

               
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
                            
                            if self.fixed_temp == True:
                                gumbel_temp = self.fixed_temp_value
                            else:
                                gumbel_temp = self.graph.update_temperature(epoch)
                            val_total_loss, _,_,_,predict_label = self.graph(cond=val_batch["sf"],label_prob=None, gumbel_temp=gumbel_temp, hard_gumbel= self.graph.hard_gumbel)

                            # loss
                            loss_val = loss_val + val_total_loss
                            n_batches = n_batches + 1        
                            
                    
                    loss_val = loss_val/n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)

                #generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0:   
                  if self.cond_model =="enc":
                        self.generator.get_cluster_performance_train(self.graph,gumbel_temp=self.fixed_temp_value, step= self.global_step)

                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                         
                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f} / Validation Loss: {loss_val:.5f}'
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
