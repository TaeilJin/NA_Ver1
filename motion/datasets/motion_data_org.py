import numpy as np
from torch.utils.data import Dataset
import glob
import os
from torch.utils.data import DataLoader
import torch

class TrainDataset_Inpainting(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, dataroot, seqlen, dropout):
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
        self.indice = []
        # load file names
        self.fnamesTrainX, self.fnamesTrainCond, self.fnamesTrainLabel = self._make_dataset(dataroot)
        #
        self.EEindex_u = [15,16,17,
                    27,28,29,
                    39,40,41 ]
        self.EEindex_l = [51,52,53,
                    63,64,65]
        #
        self.ee_HEAD_idx = [(5)*3 +0,(5)*3 +1,(5)*3 +2]
        self.ee_LH_idx=[(9)*3 +0,(9)*3 +1,(9)*3 +2]
        self.ee_RH_idx=[(13)*3 +0,(13)*3 +1,(13)*3 +2]
        self.ee_RF_idx=[(17)*3 +0,(17)*3 +1,(17)*3 +2]
        self.ee_LF_idx=[(21)*3 +0,(21)*3 +1,(21)*3 +2]
        
        self.ee_dim = 5*3 # Head, LH, RH, RF, LF 순서로 가자
        self.ee_feature = 3

        self.zeromask_ee= np.zeros((15,4),dtype=np.float32)
        self.zeromask_ee[:9,0] = True # upper end effector
        self.zeromask_ee[-6:,1] = True # lower end effector 
        self.zeromask_ee[:,2] = True # all end effector
        self.zeromask_ee[:,3] = False # all end effector is zero

        self.zeroEEcondSeq = [0,1,2,3] # All, upper only, lower only, zero

    
    def _make_dataset(self, root):

        fnamesTrainX = []
        for filename in sorted(glob.iglob(root + '/train_scaled_seqX_*.npz', recursive=True)):
            fnamesTrainX.append(filename) 
        fnamesTrainCond = []
        for filename in sorted(glob.iglob(root + '/train_scaled_seqControlAutoreg_*.npz', recursive=True)):
            fnamesTrainCond.append(filename)
        fnamesTrainLabel = []
        for filename in sorted(glob.iglob(root + '/train_scaled_seqlabel_*.npz', recursive=True)):
            fnamesTrainLabel.append(filename) 
        
        return fnamesTrainX, fnamesTrainCond, fnamesTrainLabel
                                                                                                                 
    def __len__(self):
        return len(self.fnamesTrainX)

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        # load data from batch index files
        self.x = np.load(self.fnamesTrainX[idx])['clips'].astype(np.float32)
        self.cond = np.load(self.fnamesTrainCond[idx])['clips'].astype(np.float32)
        self.label = np.load(self.fnamesTrainLabel[idx])['clips'].astype(np.int)
        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 0, 1)
        self.cond = np.swapaxes(self.cond, 0, 1)
        #self.label = np.swapaxes(self.label,0, 1)
        # # generate masked label
        self.ee_cond = np.zeros((self.ee_dim,self.x.shape[-1]),dtype=np.float32) 
        self.ee_cond[:3,:] = self.x[self.ee_HEAD_idx,:]
        self.ee_cond[(3):(3)+3,:] = self.x[self.ee_LH_idx,:]
        self.ee_cond[(6):(6)+3,:] = self.x[self.ee_RH_idx,:]
        self.ee_cond[(9):(9)+3,:] = self.x[self.ee_RF_idx,:]
        self.ee_cond[(12):(12)+3,:] = self.x[self.ee_LF_idx,:]

        if self.dropout>0.:
            n_feats, tt = self.x[:,:].shape
            cond_masked = self.cond[:,:].copy()
            
            # autoreg condition 을 사용할 때 쓴다
            keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)

            n_cond = cond_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)

            cond_masked = cond_masked*mask

            # end effector mask
            ee_cond_masked = self.ee_cond.copy()
            
            #p = np.random.rand(1)
            np.random.shuffle(self.zeroEEcondSeq)
            
            div = tt // 4

            ee_cond_p0 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)
            ee_cond_p1 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[1]:self.zeroEEcondSeq[1]+1],div,axis=-1)
            ee_cond_p2 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[2]:self.zeroEEcondSeq[2]+1],div,axis=-1)
            ee_cond_p3 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[3]:self.zeroEEcondSeq[3]+1],tt-div*3,axis=-1)
            
            ee_cond_mask = np.concatenate((ee_cond_p0,ee_cond_p1,ee_cond_p2,ee_cond_p3), axis = -1)

            ee_cond_masked = ee_cond_masked * ee_cond_mask

           
            # ee_cond_masked[:-6,:div] = 0.0 # upper
            # ee_cond_masked[-6:,div:div*2] = 0.0 # lower
            # ee_cond_masked[:,div*2:div*3] = 0.0 # lower

            sample = {'x': self.x[:,:], 'cond': cond_masked, 'ee_cond' :ee_cond_masked, 'label' : self.label}
        else:
            sample = {'x': self.x[:,:], 'cond': self.cond[:,:],'ee_cond':self.ee_cond, 'label': self.label}
            
        return sample
class ValidDataset_Inpainting(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, dataroot, seqlen, dropout):
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

        # load file names
        self.fnamesValidX, self.fnamesValidCond, self.fnamesValidLabel = self._make_dataset(dataroot)
        
        #
        self.EEindex_u = [15,16,17,
                    27,28,29,
                    39,40,41 ]
        self.EEindex_l = [51,52,53,
                    63,64,65]
                    
        #
        self.ee_HEAD_idx = [(5)*3 +0,(5)*3 +1,(5)*3 +2]
        self.ee_LH_idx=[(9)*3 +0,(9)*3 +1,(9)*3 +2]
        self.ee_RH_idx=[(13)*3 +0,(13)*3 +1,(13)*3 +2]
        self.ee_RF_idx=[(17)*3 +0,(17)*3 +1,(17)*3 +2]
        self.ee_LF_idx=[(21)*3 +0,(21)*3 +1,(21)*3 +2]
        
        self.ee_dim = 5*3 # Head, LH, RH, RF, LF 순서로 가자
        self.ee_feature = 3
        self.zeromask_ee= np.zeros((15,4),dtype=np.float32)
        self.zeromask_ee[:9,0] = True # upper end effector
        self.zeromask_ee[-6:,1] = True # lower end effector 
        self.zeromask_ee[:,2] = True # all end effector
        self.zeromask_ee[:,3] = False # all end effector is zero

        self.zeroEEcondSeq = [0,1,2,3] # All, upper only, lower only, zero
        
    
    def _make_dataset(self, root):
        fnamesValidX = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_seqX_*.npz', recursive=True)):
            fnamesValidX.append(filename) 
        fnamesValidCond = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_seqControlAutoreg_*.npz', recursive=True)):
            fnamesValidCond.append(filename)
        fnamesValidLabel = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_seqlabel_*.npz', recursive=True)):
            fnamesValidLabel.append(filename) 

        return fnamesValidX, fnamesValidCond, fnamesValidLabel                                                                     
    def __len__(self):
        return len(self.fnamesValidX)

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        # load data from batch index files
        self.x = np.load(self.fnamesValidX[idx])['clips'].astype(np.float32)
        self.cond = np.load(self.fnamesValidCond[idx])['clips'].astype(np.float32)
        self.label = np.load(self.fnamesValidLabel[idx])['clips'].astype(np.int)
        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 0, 1)
        self.cond = np.swapaxes(self.cond, 0, 1)
        #self.label = np.swapaxes(self.label,0, 1)
        # end effector
        self.ee_cond = np.zeros((self.ee_dim,self.x.shape[-1]),dtype=np.float32) 
        self.ee_cond[:3,:] = self.x[self.ee_HEAD_idx,:]
        self.ee_cond[(3):(3)+3,:] = self.x[self.ee_LH_idx,:]
        self.ee_cond[(6):(6)+3,:] = self.x[self.ee_RH_idx,:]
        self.ee_cond[(9):(9)+3,:] = self.x[self.ee_RF_idx,:]
        self.ee_cond[(12):(12)+3,:] = self.x[self.ee_LF_idx,:]
        #label
        self.label_prob = np.zeros((self.label.shape[0],3))
        if self.dropout>0.:
            n_feats, tt = self.x[:,:].shape
            cond_masked = self.cond[:,:].copy()
            
            # autoreg
            keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)

            n_cond = cond_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)

            cond_masked = cond_masked*mask

            # end effector
            ee_cond_masked = self.ee_cond.copy()
            
            div = tt // 4

            ee_cond_p0 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)
            ee_cond_p1 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[1]:self.zeroEEcondSeq[1]+1],div,axis=-1)
            ee_cond_p2 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[2]:self.zeroEEcondSeq[2]+1],div,axis=-1)
            ee_cond_p3 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[3]:self.zeroEEcondSeq[3]+1],tt-div*3,axis=-1)
            
            ee_cond_mask = np.concatenate((ee_cond_p0,ee_cond_p1,ee_cond_p2,ee_cond_p3), axis = -1)

            ee_cond_masked = ee_cond_masked * ee_cond_mask
            
            sample = {'x': self.x[:,:], 'cond': cond_masked, 'ee_cond': ee_cond_masked, 'label':self.label}
        else:
            mask_ee = np.zeros(self.x.shape)
            sample = {'x': self.x[:,:], 'cond': self.cond[:,:],'ee_cond': self.ee_cond, 'label':self.label}
            
        return sample
class TestDataset_Inpainting(Dataset):
    """Test dataset."""

    def __init__(self, dataroot, seqlen,dropout):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        """        
        self.seqlen = seqlen
        self.dropout=dropout
        self.condinfo = 3 # single control
        # load path
        self.fnamesTest, self.fnamesTest_label = self._make_dataset(dataroot)
    
    def _make_dataset(self, root):
        fnamesTest = []
        for filename in sorted(glob.glob(root + '/test_scaled_x_*.npz')):
            fnamesTest.append(filename) 
        fnamesTest_label = []
        for filename in sorted(glob.glob(root + '/test_scaled_label_*.npz')):
            fnamesTest_label.append(filename) 

        return fnamesTest, fnamesTest_label    
        
    def __len__(self):
        return len(self.fnamesTest)

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.`
        The control is not masked
        """
        # load data from batch index files
        test_data = np.load(self.fnamesTest[idx])['clips'].astype(np.float32)
        test_data_label = np.load(self.fnamesTest_label[idx])['clips'].astype(np.int)
        # Joint positions
        self.autoreg = test_data[:,:-self.condinfo] # info
        self.cond = test_data[:,-self.condinfo:]
        self.label = test_data_label[:]
        sample = {'autoreg': self.autoreg[:,:], 'cond': self.cond, 'label':self.label}
            
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
    data_root = "data/locomotion"
    train_dataset = TrainDataset_Inpainting(os.path.join(data_root,'mixamo_env_npz'),10, 0.7)
    data_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    # data_loader = get_dataloader('test', config)
    for batch in data_loader:
        # print_composite(batch)
        print(batch['x'])
