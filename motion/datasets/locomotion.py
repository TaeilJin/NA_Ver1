from opcode import hascompare
import os
import numpy as np
from .motion_data import TrainDataset_Inpainting, TestDataset_Inpainting, ValidDataset_Inpainting
from .motion_data_withFoot import TrainDataset_Inpainting_FOOT, TestDataset_Inpainting_FOOT, ValidDataset_Inpainting_FOOT
from .motion_data_NA import TrainDataset_NA,TestDataset_NA

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from visualization.plot_animation import plot_animation_withRef_tarMat, plot_animation_withRef,save_animation_BVH
import joblib
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import pdist

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled
  

class Locomotion():

    def __init__(self, hparams):
        
        data_root = hparams.Dir.data_root
        data_dir = hparams.Dir.data_dir
        testdata_dir = hparams.Dir.testdata_dir
        self.frame_rate = hparams.Data.framerate
        
        print("data loader: " + str(data_dir))
        #load data
        self.scaler =StandardScaler()
        self.scaler = joblib.load(os.path.join(data_dir,'mixamo.pkl'))

        self.seqlen = hparams.Data.seqlen
        self.x_dim = hparams.Data.pose_features
        self.env_dim = 2640
        self.sf_dim = self.x_dim*hparams.Data.seqlen + hparams.Data.target_root_features*(hparams.Data.seqlen + 1) + self.env_dim
        self.cond_dim = hparams.Data.target_root_features + self.env_dim
        print("--feature dim : " + str(hparams.Data.pose_features))
        print("--env dim : " + str(self.env_dim))
        print("--sf dim : " + str(self.sf_dim))
        print("--sf dim : " + str(self.cond_dim))

        if hparams.Train.model == "models_NA_SAMP" or "gmm_env_gmm_label_foot" or hparams.Train.model == "gmm_env_wo_gmm" or hparams.Train.model == "gmm_env_wo_gmm_woUpper" or hparams.Train.model == "gating_cVAE" or hparams.Train.model == "gmm_env_gmm_label_3part_noImpC" or hparams.Train.condmodel == "enc_foot":
            self.train_dataset = TrainDataset_NA(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.test_dataset = TestDataset_NA(testdata_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            #self.validation_dataset = ValidDataset_Inpainting_FOOT(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
        else:
            self.train_dataset = TrainDataset_Inpainting(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.test_dataset = TestDataset_Inpainting(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.validation_dataset = ValidDataset_Inpainting(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)

        self.tar_isMat = False
        self.parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1

        self.n_test = self.test_dataset.__len__()

    
    def save_animation_withRef(self, control_data, motion_data, refer_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        reference_data = np.concatenate((refer_data,control_data), axis=2)

        anim_clip = inv_standardize(animation_data, self.scaler)
        ref_clip = inv_standardize(reference_data,self.scaler)
        #np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            
            if(self.tar_isMat):
                anim_clip_input = np.concatenate((anim_clip[...,:66],anim_clip[...,-9:]),axis=-1) # get position features
                ref_clip_input = np.concatenate((ref_clip[...,:66],ref_clip[...,-9:]),axis=-1) # get position features
                plot_animation_withRef_tarMat(anim_clip_input[i,self.seqlen:,:],ref_clip_input[i,self.seqlen:,:], self.parents, filename_, fps=self.frame_rate, axis_scale=3.0)
            else:
                anim_clip_input = np.concatenate((anim_clip[...,:66],anim_clip[...,-3:]),axis=-1) # get position features
                ref_clip_input = np.concatenate((ref_clip[...,:66],ref_clip[...,-3:]),axis=-1) # get position features
                plot_animation_withRef(anim_clip_input[i,self.seqlen:,:],ref_clip_input[i,self.seqlen:,:], self.parents, filename_, fps=self.frame_rate, axis_scale=3.0)

    def save_animation_witGT(self, control_data, motion_data, refer_data, filename):
        n_clips = min(self.n_test, motion_data.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1
            anim_clip_input = np.concatenate((motion_data[...,:66],control_data[...,-3:]),axis=-1) # get position features
            ref_clip_input = np.concatenate((refer_data[...,:66],control_data[...,-3:]),axis=-1) # get position features
            plot_animation_withRef(anim_clip_input[i,self.seqlen:,:],ref_clip_input[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=3.0)
    
    
    def n_channels(self):
        return self.x_dim, self.cond_dim, self.sf_dim
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
		