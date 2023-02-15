import torch
from torch import distributions, nn
import torch.nn.functional as F
import numpy as np
import math

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, means, vars=None, device=None):
        self.n_components, self.d = means.shape
        self.means = means

        if vars is None:
            self.vars = torch.ones_like(self.means).to(device)
        else:
            self.vars = vars
        
        self.weights = torch.ones((len(means)), device=device)
        self.device = device

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, var * torch.eye(self.d).to(self.device))
                          for mean, var in zip(self.means.to(self.device), self.vars.to(self.device))]
        return gaussians

    def sample_all(self, Batches):
        all_samples = torch.cat([g.sample((Batches,)) for g in self.gaussians])
        return all_samples

    def sample_mixture_gaussians(self, all_samples, label_weight):
        
        samples = all_samples
        sample_mixtures = torch.zeros((label_weight.shape[0],samples[0].shape[-1]))
        for i in range(self.n_components):
            i_sample = samples[i].clone().detach().to(label_weight.device)
            test = torch.mm(label_weight[i:i+1] , i_sample)
            sample_mixtures[i:i+1] = test
        #sample_mixtures = torch.sum(sample_mixtures, dim=0)
        return sample_mixtures
    
    def sample_all_gaussians(self, Batches):
        all_samples = ([g.sample((Batches,)) for g in self.gaussians])
        return all_samples

    def parameters(self):
       return [self.means, self.inv_cov_std, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)
                samples[mask] = all_samples[i][mask]
        return samples

    def mixture_log_prob(self,x,label_weight):
        self.device = x.device
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1) # log N(z_i | mu_k , sigma_k)
        
        #mixture_all_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(label_weight,dim=-1)),dim=1)
        mixture_all_log_probs = torch.logsumexp(all_log_probs + torch.log(label_weight),dim=1)

        return mixture_all_log_probs

    def mixture_class_logits(self, x,label_weight):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(label_weight,dim=-1))
        return log_probs_weighted

    def classify(self, x,label_weight):
        log_probs = self.mixture_class_logits(x,label_weight)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x, label_weight):
        log_probs = self.mixture_class_logits(x,label_weight)
        return F.softmax(log_probs, dim=1)


    def log_prob(self, x, y=None, label_weight=1.): 
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights,dim=-1)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights,dim=-1))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)
