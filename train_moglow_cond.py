"""Train script.

Usage:
    train_moglow.py <hparams> <dataset>
"""
import os
import motion
import numpy as np
import datetime

from docopt import docopt
from torch.utils.data import DataLoader, Dataset
from glow.builder import build
from glow.builder_foot import build_Foot
from glow.builder_cond import build_Cond
from glow.builder_gmm import build_GMM

#from glow.trainer import Trainer
#from glow.trainer_total import Trainer_total
from glow.trainer_cond import Trainer_Cond

from glow.generator_cond import Generator_Cond
from glow.config import JsonConfig
from torch.utils.data import DataLoader
import torch
from glow.utils import save, load
if __name__ == "__main__":
    # args = docopt(__doc__)
    # hparams = args["<hparams>"]
    # dataset = args["<dataset>"]
    hparams = "hparams/preferred/locomotion.json" 
    dataset = "locomotion"

    torch.manual_seed(42)
    np.random.seed(42)
    assert dataset in motion.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, motion.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)

    dataset = hparams.Train.datasets
    print("datasets: ", dataset)
    dataset = motion.Datasets[dataset]
    
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
		
    print("log_dir:" + str(log_dir))
    print("mG_Hierarchy_Sp_EE_specify model")
    is_training = hparams.Infer.pre_trained == ""
    is_training_cond = hparams.Infer.pre_trained_cond ==""

    data = dataset(hparams, is_training)
    x_channels, cond_channels = data.n_channels()
    descriptor_channels = data.n_descriptor_channels
    # build graph
    
    print("x dim: ", x_channels)
    print("des dim: ", descriptor_channels)
    
    is_training_cond = True
    built_Cond = build_Cond(x_channels,descriptor_channels,hparams,is_training_cond)    
    
    #save(1.0,built['graph'],built['optim'],built['graph'].means)
    if is_training_cond:
        # build trainer
        if hparams.Train.condmodel == "enc":
            trainer = Trainer_Cond(**built_Cond, data=data, log_dir=log_dir, hparams=hparams)
        # train model
        trainer.train()
    else:
        generator = Generator_Cond(data, built_Cond['data_device'], log_dir, hparams)
        # generator.get_cluster_performance_train_ROT(built_Cond['graph'],gumbel_temp=0.7)

        # generator.generate_sample_recon(built_Cond['graph'],gumbel_temp=0.7)
        # generator.generate_sample_accuracy(built_Cond['graph'],gumbel_temp=0.7)
        generator.generate_sample_latent_space(built_Cond['graph'],True,0.7)
        
        


            

