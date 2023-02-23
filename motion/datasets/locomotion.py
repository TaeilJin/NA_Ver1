import os
import numpy as np
from .motion_data import TrainDataset_Inpainting, TestDataset_Inpainting, ValidDataset_Inpainting
from .motion_data_withFoot import TrainDataset_Inpainting_FOOT, TestDataset_Inpainting_FOOT, ValidDataset_Inpainting_FOOT

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from visualization.plot_animation import plot_animation, plot_animation_withRef,save_animation_BVH
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

    def __init__(self, hparams, is_training):
    
        data_root = hparams.Dir.data_root
        data_dir = hparams.Dir.data_dir
        self.frame_rate = hparams.Data.framerate
        #load data
        self.scaler =StandardScaler()
        self.scaler = joblib.load(os.path.join(data_dir,'mixamo.pkl'))

        # data root 에서 
        self.seqlen = hparams.Data.seqlen
        feature = 3
        self.joint = 22 *feature
        
        if hparams.Train.model == "gmm_env_gmm_label_foot" or hparams.Train.model == "gmm_env_wo_gmm" or hparams.Train.model == "gmm_env_wo_gmm_woUpper" or hparams.Train.model == "gating_cVAE" or hparams.Train.model == "gmm_env_gmm_label_3part_noImpC" or hparams.Train.condmodel == "enc_foot":
            self.train_dataset = TrainDataset_Inpainting_FOOT(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.test_dataset = TestDataset_Inpainting_FOOT(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.validation_dataset = ValidDataset_Inpainting_FOOT(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
        else:
            self.train_dataset = TrainDataset_Inpainting(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.test_dataset = TestDataset_Inpainting(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)
            self.validation_dataset = ValidDataset_Inpainting(data_dir,hparams.Data.seqlen, hparams.Data.dropout, hparams.Data.dropout_env)


        self.n_x_channels = self.joint
        self.env_dim = 2640
        self.n_descriptor_channels = self.n_x_channels*hparams.Data.seqlen + 3*(hparams.Data.seqlen + 1) + self.env_dim
        self.n_cond_channels = 3 + self.env_dim
        self.n_test = self.test_dataset.__len__()

    def save_APD_Score(self, control_data, K_motion_data, totalClips,filename):
        np.savez(filename + "_APD_testdata.npz", clips=K_motion_data)
        #K_motion_data = np.load("../data/results/locomotion/MG/log_20211103_1638/0_sampled_temp100_0k_APD_testdata.npz")['clips'].astype(np.float32)
        K, nn, ntimesteps, feature = K_motion_data.shape
        total_APD_score = np.zeros(nn)
        if totalClips != K:
            print("wrong! different motions")
        else :
            for nBatch in range(nn):
                k_motion_data = K_motion_data[:,nBatch,...]
                batch_control_data = control_data[nBatch:nBatch+1,...]
                k_control_data = np.repeat(batch_control_data,K,axis=0)

                apd_score = self.calculate_APD(k_control_data,k_motion_data)

                total_APD_score[nBatch] = apd_score
            print(f'APD of_{nn}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
            np.savez(filename + "_APD_score.npz", clips=total_APD_score)
            
        
    def calculate_APD(self, control_data, motion_data):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        
        motion_clip = anim_clip[...,:-3]/ 20.0

        motion_clip = np.reshape(motion_clip,(motion_clip.shape[0],-1))
        
        dist = pdist(motion_clip)

        apd = dist.mean().item()

        # #check
        # apd =0
        # n_clips = min(self.n_test, anim_clip.shape[0])
        # for i in range(0,n_clips):
        #     filename_ = f'test_{str(i)}.mp4'
        #     print('writing:' + filename_)
        #     parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
        #     plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)

        return (apd)
    
    def save_animation_UnityFile(self, motion_data, ee_data, vel_data, filename, bool_drawpy= False):
        animation_data = np.concatenate((motion_data,ee_data,vel_data), axis=-1)
                
        n_clips = min(self.n_test, animation_data.shape[0])

        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.txt'
            print('writing:' + filename_)
            parents_ee = np.array([5, 
            9,
            13,
            17,
            21]) - 1
            
            env = np.zeros((animation_data.shape[0],animation_data.shape[1],2640))
            label = np.ones((animation_data.shape[0],animation_data.shape[1],1))
            out_txt = np.concatenate((motion_data,ee_data,env,vel_data,label),axis=-1)
            np.savetxt( f'{filename}_{str(i)}.txt',out_txt[i,self.seqlen:,:],delimiter=" ")
            
    def save_animation_withRef(self, control_data, motion_data, refer_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        reference_data = np.concatenate((refer_data,control_data), axis=2)

        anim_clip = inv_standardize(animation_data, self.scaler)
        ref_clip = inv_standardize(reference_data,self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1
            anim_clip_input = np.concatenate((anim_clip[...,:66],anim_clip[...,-3:]),axis=-1) # get position features
            ref_clip_input = np.concatenate((ref_clip[...,:66],ref_clip[...,-3:]),axis=-1) # get position features
            plot_animation_withRef(anim_clip_input[i,self.seqlen:,:],ref_clip_input[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=3.0)

    def save_animation(self, control_data, motion_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=300.0)

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
        return self.n_x_channels, self.n_cond_channels
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
		