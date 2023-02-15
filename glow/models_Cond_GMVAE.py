import torch
import torch.nn as nn
from glow.Networks import *
from glow.LossFunctions import *
from glow.Metrics import *

class CondGMVAE(nn.Module):
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
        
        # gumbel
        self.init_temp = hparams.Gumbel.init_temp
        self.decay_temp = hparams.Gumbel.decay_temp
        self.hard_gumbel = hparams.Gumbel.hard_gumbel
        self.min_temp = hparams.Gumbel.min_temp
        self.decay_temp_rate = hparams.Gumbel.decay_temp_rate
        self.verbose = hparams.Gumbel.verbose
        self.gumbel_temp = self.init_temp
        
        self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
        self.losses = LossFunctions()
        self.metrics = Metrics()
    
    def unlabeled_loss(self, data, out_net):
        """Method defining the loss functions derived from the variational lower bound
        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and predictions
        """
        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec'] 
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']
        
        # reconstruction loss
        loss_rec = self.losses.reconstruction_loss(data, data_recon, self.rec_type)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(1/self.hparams.Gumbel.num_classes)

        # total loss
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

        # obtain predictions
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {'total': loss_total, 
                    'predicted_labels': predicted_labels,
                    'reconstruction': loss_rec,
                    'gaussian': loss_gauss,
                    'categorical': loss_cat}
        return loss_dic

    def update_temperature(self,epoch):
        # decay gumbel temperature
        if self.decay_temp == 1:
            self.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
        if self.verbose == 1:
          print("Gumbel Temperature: %.3lf" % self.gumbel_temp)

        return self.gumbel_temp
    def forward(self,cond=None,label_prob=None, gumbel_temp=1.0, hard_gumbel= 0):
        # data reshape 
        nBatch, nFeatures, nTimesteps = cond.shape
        cond = cond.permute(0,2,1)
        cond = cond.reshape(-1,nFeatures).clone().detach()
        if label_prob is not None:
            nBatch, nFeatures, nTimesteps = label_prob.shape
            label_prob = label_prob.permute(0,2,1).reshape(-1,nFeatures).clone().detach()


        # input 
        out_net = self.network(cond,gumbel_temp,hard_gumbel)
        if label_prob is not None:
            loss_dic = self.labeld_loss(cond, out_net, label_prob)  
        else:
            loss_dic = self.unlabeled_loss(cond,out_net)

        return loss_dic['total'], loss_dic['reconstruction'], loss_dic['gaussian'],loss_dic['categorical'], loss_dic['predicted_labels']

    def accuracy_test(self,predicted_labels,true_labels):
        # compute metrics
        accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
        nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)
        return accuracy, nmi
