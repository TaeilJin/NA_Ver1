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

class Trainer_GMM_MCFLow(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 graph_im, optim_im, lrschedule_im, loaded_step_im,
                 devices_im, data_device_im,
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

        # descriptor model
        self.graph_cond = graph_cond
        self.min_gumbel = hparams.Gumbel.fixed_temp_value
        self.hard_gumbel = hparams.Gumbel.hard_gumbel
        
        # mcflow model 
        self.graph_im = graph_im
        self.optim_im = optim_im

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

        self.loss_func = torch.nn.MSELoss(reduction='none')

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
        self.gen_model = hparams.Train.model
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
                
    def endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, loader, args):

        nf_totalloss = 0
        totalloss = 0
        total_log_loss = 0
        total_imputing = 0
        loss_func = nn.MSELoss(reduction='none')

        for index, (vectors, labels) in enumerate(loader):
            if args.use_cuda:
                vectors = vectors.cuda()
                labels[0] = labels[0].cuda()
                labels[1] = labels[1].cuda()
        
            z, nf_loss = flow.log_prob(vectors, args)
            nf_totalloss += nf_loss.item()
            z_hat = nn_model(z)
            x_hat = flow.inverse(z_hat)
            _, log_p = flow.log_prob(x_hat, args)

            batch_loss = torch.sum(loss_func(x_hat, labels[0]) * (1 - labels[1]))
            total_imputing += np.sum(1-labels[1].cpu().numpy())

            log_lss = log_p
            total_log_loss += log_p.item()
            totalloss += batch_loss.item()
            batch_loss += log_lss
            nf_loss.backward(retain_graph=True)
            nf_optimizer.step()
            nf_optimizer.zero_grad()
            batch_loss.backward()
            nn_optimizer.step()
            nn_optimizer.zero_grad()

        index+=1
        return totalloss, total_log_loss/index, nf_totalloss/index  

    def endtoend_test(flow, nn_model, data_loader, args):
        totalloss = 1
        nf_totalloss = 0
        total_imputing = 0
        loss = nn.MSELoss(reduction='none')

        for index, (vectors, labels) in enumerate(data_loader):
            if args.use_cuda:
                vectors = vectors.cuda()
                labels[0] = labels[0].cuda()
                labels[1] = labels[1].cuda()

            z, nf_loss = flow.log_prob(vectors, args)
            nf_totalloss += nf_loss.item()

            z_hat = nn_model(z)

            x_hat = flow.inverse(z_hat)

            batch_loss = torch.sum(loss(torch.clamp(x_hat, min=0, max=1), labels[0]) * labels[1])
            total_imputing += np.sum(labels[1].cpu().numpy())
            totalloss+=batch_loss.item()

        index+=1
        return totalloss/total_imputing, nf_totalloss/index

        
    def train(self):

        self.global_step = self.loaded_step
        # initial mean
        means,variance = self.calc_means(self.calc_prior,num_means = self.graph.gaussian_size, shape = self.graph.x_channels, r=1,graph_cond =self.graph_cond,train_loader=self.data_loader,device=self.data_device)
        if self.calc_prior == "mu":
            self.graph.means = means
        elif self.calc_prior == "mu_var":
            self.graph.means = means
            self.graph.vars = variance
        
        self.graph.means = torch.from_numpy(np.load('means.npz')['clips'].astype(np.float32)).to(self.data_device)
        self.graph.vars = torch.from_numpy(np.load('variance.npz')['clips'].astype(np.float32)).to(self.data_device)

        self.graph.loss_fn_GMM = SSLGaussMixture(self.graph.means, self.graph.vars)
        
        # begin to train
        for epoch in range(self.n_epoches):
            print(f"epoch:{epoch} / {self.n_epoches}")
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()
                self.graph_im.train()

                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                                                             
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                for param_group in self.optim_im.param_groups:
                    param_group['lr'] = lr

                self.optim.zero_grad()
                self.optim_im.zero_grad()

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

                # condition encoder -> prior decoding
                with torch.no_grad():
                    self.graph_cond.eval()
                    nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                    des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                    
                    out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)
                    prob = out_net['prob_cat']
                    
                    # for accuracy test "load 확인용"
                    _, predicted_labels = torch.max(out_net['logits'], dim=1)
                    _, _, nFeatures_label = label.shape
                    label_cond = label.reshape(-1,nFeatures_label).clone().detach()
                    accuracy, nmi = self.graph_cond.accuracy_test(predicted_labels.cpu().numpy(),label_cond[:,0].cpu().numpy())
                
                # init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                    self.graph_im.init_hidden()
                else:
                    self.graph.init_lstm_hidden()
                    self.graph_im.init_hidden()

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               cond[:self.batch_size // len(self.devices), ...] if cond is not None else None,
                               ee_cond[:self.batch_size // len(self.devices), ...] if ee_cond is not None else None,
                               )
                    # re-init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                        self.graph_im.init_hidden()
                    else:
                        self.graph.init_lstm_hidden()
                        self.graph_im.init_hidden()
                
                #print("n_params: " + str(self.count_parameters(self.graph)))
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                
                #   ""
                #      z, nf_loss = flow.log_prob(vectors, args)
                # nf_totalloss += nf_loss.item()
                # z_hat = nn_model(z)
                # x_hat = flow.inverse(z_hat)
                # _, log_p = flow.log_prob(x_hat, args)

                # batch_loss = torch.sum(loss_func(x_hat, labels[0]) * (1 - labels[1]))
                # total_imputing += np.sum(1-labels[1].cpu().numpy())

                # log_lss = log_p
                # total_log_loss += log_p.item()
                # totalloss += batch_loss.item()
                # batch_loss += log_lss
                # nf_loss.backward(retain_graph=True)
                # nf_optimizer.step()
                # nf_optimizer.zero_grad()
                # batch_loss.backward()
                # nn_optimizer.step()
                # nn_optimizer.zero_grad()
                #    ""

                # forward phase
                nBatch, nFeats, nTimesteps = x.shape
                # generate mask 
                masked_init = torch.zeros(x.shape).to(x.device)
                masked_init = self.graph.flow.select_layer_u.addEndEffectorElement(masked_init,ee_cond)
                masked_init[:,-24:,:] = 1
                masked_init = masked_init.bool() 
                x = x * masked_init
                
                # x -> z
                z, nll = self.graph(x=x, cond=cond, ee_cond = ee_cond)
                mult_gauss_ll_impute = self.graph.loss_multiple_gaussian(z,prob)
                mult_gauss_ll_impute = - (nll + mult_gauss_ll_impute)
                mult_gauss_ll_impute = self.graph.loss_generative(mult_gauss_ll_impute)
                #self.optim.zero_grad()
                
                #with torch.autograd.detect_anomaly():
                # imputation
                z_hat = self.graph_im(z.clone().permute(0, 2, 1)).permute(0, 2, 1)
                # hat to inverse
                x_hat = self.graph(z= z_hat, cond=cond, ee_cond=ee_cond, eps_std=1.0, reverse=True)

                #self.optim_im.zero_grad()

                # MSE Loss
                loss_ee = torch.sum(self.loss_func(x_hat, x.clone()) * (masked_init)) / nBatch

                # update flow model
                mult_gauss_ll_impute.backward(retain_graph=True)
                
                # forward phase
                z_inv_hat, nll_inv_hat = self.graph(x=x_hat, cond=cond, ee_cond = ee_cond)
                mult_gauss_ll_hat = self.graph.loss_multiple_gaussian(z_inv_hat,prob)
                mult_gauss_ll_hat = - (nll_inv_hat + mult_gauss_ll_hat)
                mult_gauss_ll_hat = self.graph.loss_generative(mult_gauss_ll_hat)
                mult_gauss_ll_hat = mult_gauss_ll_hat + loss_ee

                mult_gauss_ll_hat.backward()

                self.optim.step()
                self.optim.zero_grad()
                                
                self.optim_im.step()
                self.optim_im.zero_grad()
                    
                
                loss_total = mult_gauss_ll_impute + mult_gauss_ll_hat
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", mult_gauss_ll_impute, self.global_step)
                    self.writer.add_scalar("loss/loss_ee",loss_ee,self.global_step)
                    self.writer.add_scalar("loss/loss_total",loss_total,self.global_step)
                    

                #loss = loss_total

                # backward
                self.graph.zero_grad()
                self.graph_im.zero_grad()
                #loss.backward()

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
                    loss_generative_val =0
                    loss_ee_val =0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            self.graph_cond.eval()
                            self.graph.eval()
                            self.graph_im.eval()

                            # get validation data
                            x_val=val_batch["x"]
                            cond_val = val_batch["cond"]
                            ee_cond_val =val_batch["ee_cond"]
                            des_val = val_batch["descriptor"]
                            # calc cond_encoder
                            nBatch_cond, nFeatures_cond, nTimesteps_cond = des_val.shape
                            des_enc = des_val.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach()
                            out_net  = self.graph_cond.network(des_enc,self.min_gumbel,self.hard_gumbel)
                            prob = out_net['prob_cat']

                            # calc flow gmm loss
                            # init LSTM hidden
                            if hasattr(self.graph, "module"):
                                self.graph.module.init_lstm_hidden()
                                self.graph_im.init_hidden()
                            else:
                                self.graph.init_lstm_hidden()
                                self.graph_im.init_hidden()

                            label = val_batch["label"]
                            
                            # generate mask 
                            masked_init = torch.zeros(x_val.shape).to(z.device)
                            masked_init = self.graph.flow.select_layer_u.addEndEffectorElement(masked_init,ee_cond_val)
                            masked_init = masked_init.bool() 
                            x_val = x_val * masked_init
                            
                            # x -> z
                            z, nll = self.graph(x=x_val, cond=cond_val, ee_cond = ee_cond_val)
                            mult_gauss_ll_impute = self.graph.loss_multiple_gaussian(z,prob)
                            mult_gauss_ll_impute = - (nll + mult_gauss_ll_impute)
                            mult_gauss_ll_impute = self.graph.loss_generative(mult_gauss_ll_impute)

                            
                            # imputation
                            z_hat = self.graph_im(z.permute(0, 2, 1)).permute(0, 2, 1)
                            # hat to inverse
                            x_hat = self.graph(z= z_hat, cond=cond, ee_cond=ee_cond, eps_std=1.0, reverse=True)

                            # MSE Loss
                            val_loss_ee = torch.sum(self.loss_func(x_hat, x) * (masked_init)) / nBatch

                            # forward phase
                            z_inv_hat, nll_inv_hat = self.graph(x=x_hat, cond=cond, ee_cond = ee_cond)
                            mult_gauss_ll_hat = self.graph.loss_multiple_gaussian(z_inv_hat,prob)
                            # multiple gaussian loss
                            mult_gauss_ll_hat = - (nll_inv_hat + mult_gauss_ll_hat)
                            mult_gauss_ll_hat = self.graph.loss_generative(mult_gauss_ll_hat)
                            
                            mult_gauss_ll_hat = mult_gauss_ll_hat + val_loss_ee
                           
                            # total loss
                            loss_val = loss_val + mult_gauss_ll_impute + mult_gauss_ll_hat
                            
                            loss_generative_val = loss_generative_val + mult_gauss_ll_impute + mult_gauss_ll_hat
                            loss_ee_val = loss_ee_val + val_loss_ee

                            n_batches = n_batches + 1        
                    
                    loss_val = loss_val/n_batches
                    loss_generative_val = loss_generative_val/n_batches
                    loss_ee_val = loss_ee_val/ n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_generative_val, self.global_step)
                    self.writer.add_scalar("val_loss/val_loss_ee", loss_ee_val, self.global_step)
                    self.writer.add_scalar("val_loss/val_loss_total", loss_val, self.global_step)

                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         means=self.graph.means,
                         variance=self.graph.vars,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints,
                         b_motion=True)
                
                    save(global_step=self.global_step,
                            graph=self.graph_im,
                            optim=self.optim_im,
                            pkg_dir=self.checkpoints_dir,
                            is_best=True,
                            max_checkpoints=self.max_checkpoints,
                            b_motion=False)
                
            
                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0: 
                    if self.cond_model =="enc_rot":
                        self.generator.generate_ROT_sample_withRef_cond(self.graph,gumbel_temp=self.fixed_temp_value,step=self.global_step)
                    elif self.cond_model =="enc":
                        self.generator.generate_sample_withRef_MCFlow(self.graph,self.graph_cond,self.graph_im, eps_std=1.0, step=self.global_step)



                # global step
                self.global_step += 1
            print(
                f'Loss: {loss_total.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
