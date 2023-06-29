"""Train script.

Usage:
    train_moglow.py <hparams> <dataset>
"""
import os
import numpy as np
import datetime
import torch
from docopt import docopt
from glow.config import JsonConfig

""" action predictor """
from glow.builder_cond import build_Cond
""" pose generator """
from glow.builder_gmm import build_GMM
from glow.builder import build
from glow.builder_SAMP import build_SAMP
""" data loader """
from torch.utils.data import DataLoader, Dataset
import motion
""" train code """
from glow.trainer_cond import Trainer_Cond

from glow.trainer_GMM import Trainer_GMM
from glow.trainer_woGMM import Trainer_woGMM
from glow.trainer_moglow import Trainer_moglow
from glow.trainer_SAMP import Trainer_SAMP
""" test code """
from glow.generator import Generator
import glow.Experiment_utils as exp_utils

if __name__ == "__main__":
    # args = docopt(__doc__)
    # hparams = args["<hparams>"]
    # dataset = args["<dataset>"]
    """ fixed random seed """
    torch.manual_seed(42)
    np.random.seed(42)

    """ load hyper parameters """
    hparams = "hparams/preferred/locomotion.json" # 
    dataset = "locomotion"
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
	
    bool_main_process = True

    """ load Train & Test Datasets """
    print("results dir : " + str(log_dir))
    dataset = hparams.Train.datasets
    print("datasets name : ", dataset)
    dataset = motion.Datasets[dataset]

    print ("load data")
    data = dataset(hparams)
    x_dim, cond_dim, sf_dim = data.n_channels()
    
    """ representation """
    data.tar_isMat = hparams.Data.target_root_isMat
    if(data.tar_isMat):
        if(hparams.Data.target_root_features != 9):
            print(str(hparams.Data.target_root_features) + " you should check target is matrix representation")    
            bool_main_process = False
        
    else:
        if(hparams.Data.target_root_features != 3):
            print(str(hparams.Data.target_root_features) + " you should check target is velocity representation")
            bool_main_process= False
        
    if(hparams.Data.target_skeleton == "smpl"):
        data.parents = np.array([0,
                1,2,3,4, 
                1,6,7,8,
                1,10,11,
                12,13,14,15,
                12,17,
                12,19,20,21]) - 1
    else:
        data.parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1

    " MAIN "
    if (bool_main_process):
        """ load Network Module """
        print ("--load model")
        print("log_dir:" + str(log_dir) + "generator model: " + str(hparams.Train.model))
        is_training = hparams.Infer.pre_trained == ""
        is_training_cond = hparams.Infer.pre_trained_cond ==""
        
        print ("---action predictor")
        built_Cond = build_Cond(x_dim,sf_dim,hparams,is_training_cond)  
        
        print ("---pose generator")
        # build graph_pose
        if hparams.Train.model =="moglow":
            built = build(x_dim,cond_dim,hparams,is_training)
        elif hparams.Train.model == "gmm_env_gmm_label_history" or hparams.Train.model =="gmm_env_gmm_label_foot_history":
            built = build_GMM(x_dim, sf_dim, hparams, is_training)
        else:
            built = build_GMM(x_dim, cond_dim, hparams, is_training)

        """ training code [Action Predictor] """
        print ("---training predictor : " + str(is_training_cond))
        if is_training_cond:
            # build trainer
            if hparams.Train.condmodel == "enc":
                trainer = Trainer_Cond(**built_Cond, data=data, log_dir=log_dir, hparams=hparams)
            # train model
            trainer.train()

        """ training code [Motion Generator] """
        if is_training:
            if hparams.Train.model=="moglow":
                trainer = Trainer_moglow(**built, data=data, log_dir=log_dir, hparams=hparams)
            if hparams.Train.model=="samp":
                trainer = Trainer_SAMP(**built, data=data, log_dir=log_dir, hparams=hparams)
            if hparams.Train.model =="models_NA_SAMP" or "gmm_env_wo_gmm" or hparams.Train.model =="gmm_env_wo_gmm_woUpper" or hparams.Train.model =="gmm_env_gmm_label":
                trainer = Trainer_woGMM(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
            
            # train model
            print("descriptor params : " ,trainer.count_parameters(built_Cond['graph']))
            print("generator params : " ,trainer.count_parameters(built['graph']))
            trainer.train()
        
