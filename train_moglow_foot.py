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
from glow.builder_total import build_total
from glow.builder_cond import build_Cond
from glow.builder_gmm import build_GMM
from glow.builder_foot import build_Foot

from glow.trainer_cond_Foot import Trainer_Foot_Estimator

from glow.generator import Generator
#from glow.generator_cond import Generator_Cond
from glow.config import JsonConfig
from torch.utils.data import DataLoader
import torch
from glow.utils import save, load
if __name__ == "__main__":
    #args = docopt(__doc__)
    #hparams = args["<hparams>"]
    #dataset = args["<dataset>"]
    hparams = "hparams/preferred/locomotion.json" # 
    dataset = "locomotion"

    torch.manual_seed(42)

    assert dataset in motion.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, motion.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = motion.Datasets[dataset]
    
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
	
    
    dataset = hparams.Train.datasets
    print("datasets: ", dataset)
    dataset = motion.Datasets[dataset]

    print("log_dir:" + str(log_dir))
    print("mG_Hierarchy_Sp_EE_specify model")
    is_training = hparams.Infer.pre_trained == ""
    is_training_cond = hparams.Infer.pre_trained_cond ==""
    is_training_foot = hparams.Infer.pre_trained_foot ==""
    
    data = dataset(hparams, is_training)
    x_channels, cond_channels = data.n_channels()
    descriptor_channels = data.n_descriptor_channels
    # build graph
    built_Cond = build_Cond(x_channels,descriptor_channels,hparams,is_training_cond)  
    built = build_GMM(x_channels, cond_channels, hparams, is_training)
        
    # build foot contact
    if hparams.Train.footmodel == "gating" or hparams.Train.footmodel == "concat":
        built_Foot = build_Foot(x_channels,descriptor_channels,hparams,is_training_foot)
        is_training = is_training_foot
        for name, param in built_Foot['graph'].named_parameters():
            if param.requires_grad:
                print(name)
                #print(param.data)
    
    
    if is_training_foot:
        # build trainer
        trainer = Trainer_Foot_Estimator(**built_Foot, graph_cond=built_Cond['graph'],graph_pose=built['graph'],data=data,log_dir=log_dir,hparams=hparams)
        # train model
        trainer.train()
    else:
        #generator = Generator_Cond(data, built_Cond['data_device'], log_dir, hparams)
        #generator.generate_sample_recon(built_Cond['graph'],gumbel_temp=1.0)
        #generator.generate_sample_accuracy(built_Cond['graph'],gumbel_temp=1.0)
        #generator.generate_sample_latent_sapce(built_Cond['graph'],True,1.0)
        
        
        
        # Synthesize a lot of data. 
        generator = Generator(data, built['data_device'], log_dir, hparams)
        if "temperature" in hparams.Infer:
            temp = hparams.Infer.temperature
        else:
            temp = 1
            
        # We generate x times to get some different variations for each input
        if hparams.Train.model=="hg":
            for i in range(1):            
                generator.generate_diverse_motion_withDlow(built_dlow['graph'],built['graph'],built_Cond['graph'],eps_std=temp)
        if hparams.Train.model=="gmm":
            #generator.generate_diverse_motion(built['graph'],built_Cond['graph'],hparams.Gumbel.num_classes)
            for i in range(1):            
                generator.generate_diverse_motion_withDlow(built_dlow['graph'],built['graph'],built_Cond['graph'],eps_std=temp)
       

            

