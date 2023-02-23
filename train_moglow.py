"""Train script.

Usage:
    train_moglow.py <hparams> <dataset>
"""
import os
from glow.trainer_GMM_label_history import Trainer_Label_HISTORY

import motion
import numpy as np
import datetime

from docopt import docopt
from torch.utils.data import DataLoader, Dataset
from glow.builder import build
from glow.builder_cond import build_Cond
from glow.builder_gmm import build_GMM
from glow.builder_foot import build_Foot
from glow.builder_imputing import build_Imputing
from glow.builder_SAMP import build_SAMP 

from glow.trainer_cond import Trainer_Cond
from glow.trainer_GMM import Trainer_GMM
from glow.trainer_GMM_UPPER import Trainer_GMM_UPPER
from glow.trainer_woGMM import Trainer_woGMM
from glow.trainer_GMM_label import Trainer_Label
from glow.trainer_GMM_label_foot import Trainer_Label_Foot
from glow.trainer_GMM_label_history_foot import Trainer_Label_HISTORY_Foot
from glow.trainer_moglow import Trainer_moglow
from glow.trainer_SAMP import Trainer_SAMP

from glow.generator import Generator
#from glow.generator_cond import Generator_Cond
from glow.config import JsonConfig
from torch.utils.data import DataLoader
import torch
from glow.utils import save, load
import glow.Experiment_utils as exp_utils

if __name__ == "__main__":
    # args = docopt(__doc__)
    # hparams = args["<hparams>"]
    # dataset = args["<dataset>"]
    torch.manual_seed(42)
    np.random.seed(42)

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
	
    dataset = hparams.Train.datasets
    print("datasets: ", dataset)
    dataset = motion.Datasets[dataset]
    

    print("log_dir:" + str(log_dir))
    print("mG_Hierarchy_Sp_EE_specify model")
    is_training = hparams.Infer.pre_trained == ""
    is_training_cond = hparams.Infer.pre_trained_cond ==""
     

    data = dataset(hparams, is_training)
    x_channels, cond_channels = data.n_channels()
    descriptor_channels = data.n_descriptor_channels
    
    
    # build graph
    if hparams.Train.model=="samp":
        built_Cond =None
    else:
        built_Cond = build_Cond(x_channels,descriptor_channels,hparams,is_training_cond)  
    
    # build graph_pose
    if hparams.Train.model =="moglow":
        built = build(x_channels,cond_channels,hparams,is_training)
    elif hparams.Train.model == "gmm_env_gmm_label_history" or hparams.Train.model =="gmm_env_gmm_label_foot_history":
        built = build_GMM(x_channels, descriptor_channels, hparams, is_training)
    else:
        built = build_GMM(x_channels, cond_channels, hparams, is_training)

    
    
    if is_training:
        # build trainer
        if hparams.Train.model=="moglow":
            trainer = Trainer_moglow(**built, data=data, log_dir=log_dir, hparams=hparams)
        if hparams.Train.model=="gmm" or hparams.Train.model=="gmm_env" or hparams.Train.model=="gmm_env_wo" or  hparams.Train.model=="gmm_env_noEE" or hparams.Train.model =="gmm_env_with_label":
            trainer = Trainer_GMM(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if hparams.Train.model =="gmm_env_wo_gmm" or hparams.Train.model =="gmm_env_wo_gmm_woUpper" or hparams.Train.model =="gmm_env_gmm_label":
            trainer = Trainer_woGMM(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if hparams.Train.model=="gmm_env_wo_upper":
            trainer = Trainer_GMM_UPPER(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if  hparams.Train.model =="gmm_env_gmm_label":
            trainer = Trainer_Label(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if  hparams.Train.model == "gmm_env_gmm_label_history":
            trainer = Trainer_Label_HISTORY(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if  hparams.Train.model == "gmm_env_gmm_label_foot_history":
            trainer = Trainer_Label_HISTORY_Foot(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if hparams.Train.model == "gmm_env_gmm_label_foot" or hparams.Train.model == "gmm_env_gmm_label_foot_3part" or hparams.Train.model == "gmm_env_gmm_label_3part" or hparams.Train.model == "gmm_env_gmm_label_3part_noImpC":
            trainer = Trainer_Label_Foot(**built, graph_cond=built_Cond['graph'], data=data, log_dir=log_dir, hparams=hparams)
        if hparams.Train.model=="samp":
            trainer = Trainer_SAMP(**built, data=data, log_dir=log_dir, hparams=hparams)

        # train model
        print("descriptor params : " ,trainer.count_parameters(built_Cond['graph']))
        print("generator params : " ,trainer.count_parameters(built['graph']))
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
                generator.generate_sample_withRef(built['graph'],eps_std=temp, counter=i)
        if hparams.Train.model=="gmm":
            for i in range(1):   
                generator.generate_0515_sample_withRef("Demo_lc_ld_b.txt", hparams.Dir.data_dir, built['graph'],built_Cond['graph'],eps_std=temp, counter=i)          
                #generator.generate_sample_withRef_cond(built['graph'],built_Cond['graph'],eps_std=temp, counter=i)
        if hparams.Train.model =="moglow":
            data = np.load("/root/home/project/Data/root_position_all/experiments/trans_lcw_6.txt.npz")['clips'].astype(np.float32)
            data = np.array(data,ndmin=3).astype(np.float32)
            autoreg_all = data[:,:,:66]
            autoreg_all = exp_utils.Normalize_motion(autoreg_all,generator.data.scaler).astype(np.float32)
            control_all = data[:,:,-4:-1]
            control_all = exp_utils.Normalize_vel(control_all,generator.data.scaler).astype(np.float32)
            env_all = data[:,:,66+30:-4] 

            generator.generate_sample_withRef_cond_moglow(built['graph'], eps_std=1.0)
            #generator.generate_sample_withRef(built['graph'], eps_std=1.0, counter=0,autoreg_all=autoreg_all,control_all=control_all,env_all=env_all)
        
        
        if hparams.Train.model =="gmm_env"or hparams.Train.model =="gmm_env_wo_gmm" or hparams.Train.model =="gmm_env_wo_upper" or hparams.Train.model=="gmm_env_gmm_label":
            
            autoreg_all = generator.test_batch["autoreg"].cpu().numpy()
            control_all = generator.test_batch["cond"].cpu().numpy()
            env_all = generator.test_batch["env"].cpu().numpy()
            #
            data = np.load("/root/home/project/Data/root_position_all/experiments/loco_sliced.npz")['clips'].astype(np.float32)
            autoreg_all = data[:,:,:66]
            control_all = data[:,:,-4:-1]
            env_all = data[:,:,66+30:-4] 

            data = np.load("/root/home/project/Data/root_position_all/experiments/env_sliced.npz")['clips'].astype(np.float32)
            autoreg_all_env = data[:,:,:66]
            autoreg_all_env = exp_utils.Normalize_motion(autoreg_all_env,generator.data.scaler).astype(np.float32)
            control_all_env = data[:,:,-4:-1]
            control_all_env = exp_utils.Normalize_vel(control_all_env,generator.data.scaler).astype(np.float32)
            env_all_env = data[:,:,66+30:-4] 

            generator.generate_test_eval(built['graph'], built_Cond['graph'], seqlen=10, test_gt=autoreg_all_env, test_cond=control_all_env, test_env=env_all_env)
            #generator.generate_test_eval_noGMM(built['graph'], built_Cond['graph'], seqlen=10, test_gt=autoreg_all_env, test_cond=control_all_env, test_env=env_all_env)
            #generator.generate_test_label(built['graph'], built_Cond['graph'], seqlen=10, test_gt=autoreg_all_env, test_cond=control_all_env, test_env=env_all_env)
            



        if hparams.Train.model=="gmm_rot":
            #generator.generate_ROT_0515_sample_withRef("Demo_lc_ld_b.txt", hparams.Dir.data_dir, built['graph'],built_Cond['graph'],eps_std=temp, counter=0)   
            generator.generate_ROT_sample_withRef_cond(built['graph'],built_Cond['graph'],eps_std=temp, counter=0) 
        
        if hparams.Train.model=="fix_cond":
            for i in range(5):            
                generator.generate_sample_withRef_cond_enc(built['graph'],built_Cond['graph'],eps_std=temp, counter=i)
        if hparams.Train.model=="total":
            for i in range(5):            
                generator.generate_sample_withRef_cond_enc(built['model'].graph,built['model'].graph_cond,eps_std=temp, counter=i)
        if hparams.Train.model=="samp":
            for i in range(1):  
                generator.generate_sample_withRef_SAMP(built['graph'],eps_std=1.0, counter=i)

            

