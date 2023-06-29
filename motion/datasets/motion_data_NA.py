import enum
import numpy as np
from torch.utils.data import Dataset
import glob
import os
from torch.utils.data import DataLoader
import torch

class TrainDataset_NA(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, dataroot, seqlen, dropout, dropout_env):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        seqlen: number of autoregressive body poses and previous control values
        n_lookahead: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.seqlen = seqlen
        self.dropout=dropout
        self.dropout_env = dropout_env
        self.indice = []
        # load file names
        self.fnamesTrainX, self.fnamesTrainCond, self.fnamesTrainSceneFeatures, self.fnamesTrainEnv, self.fnamesTrainEE = self._make_dataset(dataroot)
        
        # for drop condition
        self.zeromask_ee= np.zeros((15,4),dtype=np.float32)
        self.zeromask_ee[:9,0] = True # upper end effector
        self.zeromask_ee[:9,1] = True # upper end effector 
        self.zeromask_ee[:,2] = True # all end effector
        self.zeromask_ee[:,3] = False # all end effector is zero

        self.zeroEEcondSeq = [0,1,2,3] # All, upper only, lower only, zero

    
    def _make_dataset(self, root):

        fnamesTrainX = []
        for filename in sorted(glob.iglob(root + '/train_scaled_x_*.npz', recursive=True)):
            fnamesTrainX.append(filename) 
        fnamesTrainCond = []
        for filename in sorted(glob.iglob(root + '/train_scaled_singleControl_*.npz', recursive=True)):
            fnamesTrainCond.append(filename)
        fnamesTrainEnv = []
        for filename in sorted(glob.iglob(root + '/train_scaled_env_*.npz', recursive=True)):
            fnamesTrainEnv.append(filename)
        fnamesTrainSceneFeatures = []
        for filename in sorted(glob.iglob(root + '/train_scaled_seqControlAutoreg_*.npz', recursive=True)):
            fnamesTrainSceneFeatures.append(filename)
        fnamesTrainEE = []
        for filename in sorted(glob.iglob(root + '/train_scaled_ee_*.npz', recursive=True)):
            fnamesTrainEE.append(filename)
        
        return fnamesTrainX, fnamesTrainCond, fnamesTrainSceneFeatures, fnamesTrainEnv,fnamesTrainEE
                                                                                                                 
    def __len__(self):
        return len(self.fnamesTrainX)

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        # load data from batch index files
        self.x = np.load(self.fnamesTrainX[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        self.cond = np.load(self.fnamesTrainCond[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        self.sf = np.load(self.fnamesTrainSceneFeatures[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        #202207
        self.env = np.load(self.fnamesTrainEnv[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        
        self.ee_cond = np.load(self.fnamesTrainEE[idx])['clips'].astype(np.float32).swapaxes(0, 1)
     
        """ data masking """
        n_feats, tt = self.x[:,:].shape
        # environment 
        des_env = self.env.copy()
        keep_env = np.random.rand(1,tt) < (1- self.dropout_env)
        mask_env = np.repeat(keep_env, des_env.shape[0], axis = 0)
        des_env = np.where((des_env > 1e-3) & (mask_env == True),des_env,0.0)
      
        # descriptor (sequence + env)
        sf_masked = self.sf[:,:].copy()
        keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)
        n_cond = sf_masked.shape[0]-(n_feats*self.seqlen)
        mask_cond = np.full((n_cond, tt), True)
        mask = np.repeat(keep_pose, n_feats, axis = 0)
        mask = np.concatenate((mask, mask_cond), axis=0)
        sf_masked = sf_masked*mask     
        sf_masked = np.concatenate((sf_masked, des_env),axis=0) # dropouted descriptors       

        # condition
        cond_masked = self.cond[:,:].copy()
        cond_masked = np.concatenate((cond_masked,des_env),axis=0) # dropouted condition (vel + env) 

        # end effector mask
        ee_cond_masked = self.ee_cond.copy()
        np.random.shuffle(self.zeroEEcondSeq)
        div = tt // 4
        # end-effector 를 batch 에서 frame 마다 random 하게 줘봤다
        ee_cond_p0 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)
        ee_cond_p1 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[1]:self.zeroEEcondSeq[1]+1],div,axis=-1)
        ee_cond_p2 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[2]:self.zeroEEcondSeq[2]+1],div,axis=-1)
        ee_cond_p3 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[3]:self.zeroEEcondSeq[3]+1],tt-div*3,axis=-1)
        ee_cond_mask = np.concatenate((ee_cond_p0,ee_cond_p1,ee_cond_p2,ee_cond_p3), axis = -1)
        ee_cond_masked = ee_cond_masked * ee_cond_mask

        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        sample = {'x': self.x[:,:], 'cond': cond_masked, 'ee_gt':self.ee_cond, 'ee_cond' :ee_cond_masked, 'sf' : sf_masked }
            
        return sample

class TestDataset_NA(Dataset):
    """Test dataset."""

    def __init__(self, dataroot, seqlen,dropout, dropout_env):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        """        
        self.seqlen = seqlen
        self.dropout=dropout
        self.dropout_env = dropout_env
        self.condinfo = 3 # single control
        # load path
        self.fnamesTest , self.fnamesTestCond, self.fnamesTestSceneFeatures, self.fnamesTestEnv, self.fnamesTestEE = self._make_dataset(dataroot)

        # for drop condition
        self.zeromask_ee= np.zeros((15,4),dtype=np.float32)
        self.zeromask_ee[:9,0] = True # upper end effector
        self.zeromask_ee[:9,1] = True # upper end effector 
        self.zeromask_ee[:,2] = True # all end effector
        self.zeromask_ee[:,3] = False # all end effector is zero

        self.zeroEEcondSeq = [0,1,2,3] # All, upper only, lower only, zero

    def _make_dataset(self, root):
        fnamesTestX = []
        for filename in sorted(glob.iglob(root + '/test_scaled_x_*.npz', recursive=True)):
            fnamesTestX.append(filename) 
        fnamesTestCond = []
        for filename in sorted(glob.iglob(root + '/test_scaled_singleControl_*.npz', recursive=True)):
            fnamesTestCond.append(filename)
        fnamesTestEnv = []
        for filename in sorted(glob.iglob(root + '/test_scaled_env_*.npz', recursive=True)):
            fnamesTestEnv.append(filename)
        fnamesTestSceneFeatures = []
        for filename in sorted(glob.iglob(root + '/test_scaled_seqControlAutoreg_*.npz', recursive=True)):
            fnamesTestSceneFeatures.append(filename)
        fnamesTestEE = []
        for filename in sorted(glob.iglob(root + '/test_scaled_ee_*.npz', recursive=True)):
            fnamesTestEE.append(filename)

        return fnamesTestX, fnamesTestCond, fnamesTestSceneFeatures, fnamesTestEnv,fnamesTestEE
        
    def __len__(self):
        return len(self.fnamesTest)

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        # load data from batch index files
        self.x = np.load(self.fnamesTest[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        self.cond = np.load(self.fnamesTestCond[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        self.sf = np.load(self.fnamesTestSceneFeatures[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        #202207
        self.env = np.load(self.fnamesTestEnv[idx])['clips'].astype(np.float32).swapaxes(0, 1)
        
        self.ee_cond = np.load(self.fnamesTestEE[idx])['clips'].astype(np.float32).swapaxes(0, 1)
     
        """ data masking """
        n_feats, tt = self.x[:,:].shape
        # environment 
        des_env = self.env.copy()
        keep_env = np.random.rand(1,tt) < (1- self.dropout_env)
        mask_env = np.repeat(keep_env, des_env.shape[0], axis = 0)
        des_env = np.where((des_env > 1e-3) & (mask_env == True),des_env,0.0)
      
        # descriptor (sequence + env)
        sf_masked = self.sf[:,:].copy()
        keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)
        n_cond = sf_masked.shape[0]-(n_feats*self.seqlen)
        mask_cond = np.full((n_cond, tt), True)
        mask = np.repeat(keep_pose, n_feats, axis = 0)
        mask = np.concatenate((mask, mask_cond), axis=0)
        sf_masked = sf_masked*mask     
        sf_masked = np.concatenate((sf_masked, des_env),axis=0) # dropouted descriptors       

        # condition
        cond_masked = self.cond[:,:].copy()
        cond_masked = np.concatenate((cond_masked,des_env),axis=0) # dropouted condition (vel + env) 

        # end effector mask
        ee_cond_masked = self.ee_cond.copy()
        np.random.shuffle(self.zeroEEcondSeq)
        div = tt // 4
        # end-effector 를 batch 에서 frame 마다 random 하게 줘봤다
        ee_cond_p0 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)
        ee_cond_p1 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[1]:self.zeroEEcondSeq[1]+1],div,axis=-1)
        ee_cond_p2 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[2]:self.zeroEEcondSeq[2]+1],div,axis=-1)
        ee_cond_p3 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[3]:self.zeroEEcondSeq[3]+1],tt-div*3,axis=-1)
        ee_cond_mask = np.concatenate((ee_cond_p0,ee_cond_p1,ee_cond_p2,ee_cond_p3), axis = -1)
        ee_cond_masked = ee_cond_masked * ee_cond_mask

        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        sample = {'x': self.x[:,:], 'cond_gt':self.cond, 'env_gt': self.env, 'cond': cond_masked, 'ee_gt':self.ee_cond, 'ee_cond' :ee_cond_masked, 'sf' : sf_masked }    
        return sample



def print_composite(data, beg=""):
    if isinstance(data, dict):
        print(f'{beg} dict, size = {len(data)}')
        for key, value in data.items():
            print(f'  {beg}{key}:')
            print_composite(value, beg + "    ")
    elif isinstance(data, list):
        print(f'{beg} list, len = {len(data)}')
        for i, item in enumerate(data):
            print(f'  {beg}item {i}')
            print_composite(item, beg + "    ")
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f'{beg} array of size {data.shape}')
    else:
        print(f'{beg} {data}')
if __name__ == '__main__':
    data_root = "/root/home/project/mnt/ssd23/TAEIL/SAMP/Train/npz"
    train_dataset = TrainDataset_NA(data_root,10, 0.7, 0.1)
    data_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    print("-----train data information------")
    for i, batch in enumerate(data_loader) :
        print_composite(batch)
        if(i==0):
            break;
    
    data_root = "/root/home/project/mnt/ssd23/TAEIL/SAMP/Test/npz"
    test_dataset = TestDataset_NA(data_root,10, 0.7, 0.1)
    data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    print("-----test data information-----")
    for i, batch in enumerate(data_loader) :
        print_composite(batch)
        if(i==0):
            break;
