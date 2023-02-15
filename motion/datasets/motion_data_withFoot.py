import numpy as np
from torch.utils.data import Dataset
import glob
import os
from torch.utils.data import DataLoader
import torch

class TrainDataset_Inpainting_FOOT(Dataset):
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
        self.fnamesTrainX, self.fnamesTrainCond, self.fnamesTrainDescriptor, self.fnamesTrainLabel, self.fnamesTrainEnv, self.fnamesTrainFoot = self._make_dataset(dataroot)
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
        # for drop condition
        self.zeromask_ee= np.zeros((15,4),dtype=np.float32)
        self.zeromask_ee[:9,0] = True # upper end effector
        self.zeromask_ee[9:,1] = True # lower end effector 
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
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        fnamesTrainDescriptor = []
        for filename in sorted(glob.iglob(root + '/train_scaled_seqControlAutoreg_*.npz', recursive=True)):
            fnamesTrainDescriptor.append(filename)
        # 20220419. label 을 얻었다.
        fnamesTrainLabel = []
        for filename in sorted(glob.iglob(root + '/train_scaled_label_*.npz', recursive=True)):
            fnamesTrainLabel.append(filename) 
        # 20221201. foot 을 얻었다.
        fnamesTrainFoot = []
        for filename in sorted(glob.iglob(root + '/train_scaled_fcontact_*.npz', recursive=True)):
            fnamesTrainFoot.append(filename) 

        return fnamesTrainX, fnamesTrainCond, fnamesTrainDescriptor, fnamesTrainLabel, fnamesTrainEnv,fnamesTrainFoot
                                                                                                                 
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
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        self.descriptor = np.load(self.fnamesTrainDescriptor[idx])['clips'].astype(np.float32)
        # 20220419. label 을 얻는다.
        self.label = np.load(self.fnamesTrainLabel[idx])['clips'].astype(np.int)
        #202207
        self.env = np.load(self.fnamesTrainEnv[idx])['clips'].astype(np.float32)
        #202212
        self.foot = np.load(self.fnamesTrainFoot[idx])['clips'].astype(np.float32)

        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 0, 1)
        self.cond = np.swapaxes(self.cond, 0, 1)
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        self.descriptor = np.swapaxes(self.descriptor,0,1)    
        # 202207
        self.env = np.swapaxes((self.env), 0, 1)
        # 20221201
        self.foot = np.swapaxes(self.foot, 0, 1)

        # # generate masked label
        self.ee_cond = np.zeros((self.ee_dim,self.x.shape[-1]),dtype=np.float32) 
        self.ee_cond[:3,:] = self.x[self.ee_HEAD_idx,:]
        self.ee_cond[(3):(3)+3,:] = self.x[self.ee_LH_idx,:]
        self.ee_cond[(6):(6)+3,:] = self.x[self.ee_RH_idx,:]
        self.ee_cond[(9):(9)+3,:] = self.x[self.ee_RF_idx,:]
        self.ee_cond[(12):(12)+3,:] = self.x[self.ee_LF_idx,:]

        if self.dropout is not None:
            n_feats, tt = self.x[:,:].shape
            cond_masked = self.cond[:,:].copy()
            
            
            # descriptor (sequence + env)
            des_masked = self.descriptor[:,:].copy()

            # autoreg condition 을 사용할 때 쓴다
            keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)

            n_cond = des_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)

            des_masked = des_masked*mask

            # environment 
            des_env = self.env.copy()
            # environment dropout
            keep_env = np.random.rand(1,tt) < (1- self.dropout_env)
            mask_env = np.repeat(keep_env, des_env.shape[0], axis = 0)
            des_env = np.where((des_env > 1e-3) & (mask_env == True),des_env,0.0)
            
            # dropouted descriptors            
            des_masked = np.concatenate((des_masked, des_env),axis=0)

            # dropouted condition (vel + env) 
            cond_masked = np.concatenate((cond_masked,des_env),axis=0)

            # end effector mask
            ee_cond_masked = self.ee_cond.copy()
            
            np.random.shuffle(self.zeroEEcondSeq)
            
            # div = tt // 4
            # # end-effector 를 batch 에서 frame 마다 random 하게 줘봤다
            # ee_cond_p0 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)
            # ee_cond_p1 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[1]:self.zeroEEcondSeq[1]+1],div,axis=-1)
            # ee_cond_p2 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[2]:self.zeroEEcondSeq[2]+1],div,axis=-1)
            # ee_cond_p3 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[3]:self.zeroEEcondSeq[3]+1],tt-div*3,axis=-1)
            # ee_cond_mask = np.concatenate((ee_cond_p0,ee_cond_p1,ee_cond_p2,ee_cond_p3), axis = -1)
            
            div = tt
            ee_cond_mask = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)

            ee_cond_masked = ee_cond_masked * ee_cond_mask

            # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
            sample = {'x': self.x[:,:], 'cond': cond_masked, 'ee_gt':self.ee_cond, 'ee_cond' :ee_cond_masked, 'descriptor' : des_masked, 'label':self.label[:,:], 'foot':self.foot}
        # else:
        #     sample = {'x': self.x[:,:], 'cond': self.cond[:,:],'ee_cond':self.ee_cond,'descriptor' : self.descriptor[:,:],'label':self.label[:,:],'foot':self.foot}
            
        return sample
class ValidDataset_Inpainting_FOOT(Dataset):
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
        # load file names
        # 20220419. label 을 넣었다  Valid
        self.fnamesValidX, self.fnamesValidCond, self.fnamesValidDescriptor, self.fnamesValidLabel, self.fnamesValidEnv, self.fnamesValidFoot = self._make_dataset(dataroot)
        
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

        # for drop condition
        self.zeromask_ee= np.zeros((15,4),dtype=np.float32)
        self.zeromask_ee[:9,0] = True # upper end effector
        self.zeromask_ee[9:,1] = True # upper end effector 
        self.zeromask_ee[:,2] = True # all end effector
        self.zeromask_ee[:,3] = False # all end effector is zero

        self.zeroEEcondSeq = [0,1,2,3] # All, upper only, lower only, zero

    
    def _make_dataset(self, root):
        fnamesValidX = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_x_*.npz', recursive=True)):
            fnamesValidX.append(filename) 
        fnamesValidCond = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_singleControl_*.npz', recursive=True)):
            fnamesValidCond.append(filename)
        
        fnamesValidEnv =[]
        for filename in sorted(glob.iglob(root + '/valid_scaled_env_*.npz', recursive=True)):
            fnamesValidEnv.append(filename)
        
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다. Valid
        fnamesValidDescriptor = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_seqControlAutoreg_*.npz', recursive=True)):
            fnamesValidDescriptor.append(filename)
        # 20220419. label 을 얻는다. label
        fnamesValidLabel = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_label_*.npz', recursive=True)):
            fnamesValidLabel.append(filename)
        # 202212-1 foot 얻는다.
        fnamesValidFoot = []
        for filename in sorted(glob.iglob(root + '/valid_scaled_fcontact_*.npz', recursive=True)):
            fnamesValidFoot.append(filename)

        return fnamesValidX, fnamesValidCond, fnamesValidDescriptor, fnamesValidLabel, fnamesValidEnv,fnamesValidFoot                                                                     
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
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다. Valid
        self.descriptor = np.load(self.fnamesValidDescriptor[idx])['clips'].astype(np.float32)
        # 20220419. label 얻기
        self.label = np.load(self.fnamesValidLabel[idx])['clips'].astype(np.int)
        # env
        self.env = np.load(self.fnamesValidEnv[idx])['clips'].astype(np.float32)
        # 20221201 foot 얻기
        self.foot = np.load(self.fnamesValidFoot[idx])['clips'].astype(np.float32)

        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 0, 1)
        self.cond = np.swapaxes(self.cond, 0, 1)
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다. Valid
        self.descriptor = np.swapaxes(self.descriptor, 0, 1)
        #
        self.env = np.swapaxes(self.env, 0, 1)
        # 20221201
        self.foot = np.swapaxes(self.foot, 0, 1)
        
        # end effector
        self.ee_cond = np.zeros((self.ee_dim,self.x.shape[-1]),dtype=np.float32) 
        self.ee_cond[:3,:] = self.x[self.ee_HEAD_idx,:]
        self.ee_cond[(3):(3)+3,:] = self.x[self.ee_LH_idx,:]
        self.ee_cond[(6):(6)+3,:] = self.x[self.ee_RH_idx,:]
        self.ee_cond[(9):(9)+3,:] = self.x[self.ee_RF_idx,:]
        self.ee_cond[(12):(12)+3,:] = self.x[self.ee_LF_idx,:]

        if self.dropout is not None:
            n_feats, tt = self.x[:,:].shape
            cond_masked = self.cond[:,:].copy()
            # descriptor
            des_masked = self.descriptor.copy()
            
            # autoreg condition 을 사용할 때 쓴다
            keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)

            n_cond = des_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)

            des_masked = des_masked*mask

            # environment 
            des_env = self.env.copy()
            # environment dropout
            keep_env = np.random.rand(1,tt) < (1- self.dropout_env)
            mask_env = np.repeat(keep_env, des_env.shape[0], axis = 0)
            des_env = np.where((des_env > 1e-3) & (mask_env == True),des_env,0.0)

            # dropout-ed descriptor
            des_masked = np.concatenate((des_masked, des_env),axis=0)

            # dropout-ed condition (vel + env) 
            cond_masked = np.concatenate((cond_masked,des_env),axis=0)
            
            # end effector mask
            ee_cond_masked = self.ee_cond.copy()
            
            np.random.shuffle(self.zeroEEcondSeq)
            
            #div = tt // 4
            #ee_cond_p0 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)
            #ee_cond_p1 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[1]:self.zeroEEcondSeq[1]+1],div,axis=-1)
            #ee_cond_p2 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[2]:self.zeroEEcondSeq[2]+1],div,axis=-1)
            #ee_cond_p3 = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[3]:self.zeroEEcondSeq[3]+1],tt-div*3,axis=-1)
            #ee_cond_mask = np.concatenate((ee_cond_p0,ee_cond_p1,ee_cond_p2,ee_cond_p3), axis = -1)
            div = tt
            ee_cond_mask = np.repeat(self.zeromask_ee[:,self.zeroEEcondSeq[0]:self.zeroEEcondSeq[0]+1],div,axis=-1)

            ee_cond_masked = ee_cond_masked * ee_cond_mask
            # 20220419 valid 데이터를 내보낸다.
            sample = {'x': self.x[:,:], 'cond': cond_masked, 'ee_gt':self.ee_cond, 'ee_cond': ee_cond_masked, 'descriptor':des_masked, 'label':self.label, 'foot':self.foot}
        else:
            sample = {'x': self.x[:,:], 'cond': self.cond[:,:],'ee_cond': self.ee_cond, 'descriptor':self.descriptor[:,:], 'label':self.label, 'foot':self.foot}
            
        return sample
class TestDataset_Inpainting_FOOT(Dataset):
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
        self.fnamesTest , self.fnamesTest_label, self.fnamesTest_Vel, self.fnamesTest_Env, self.fnamesTest_EE, self.fnamesTest_Foot = self._make_dataset(dataroot)
    
    def _make_dataset(self, root):
        fnamesTest = []
        for filename in sorted(glob.glob(root + '/test_scaled_x_*.npz')):
            fnamesTest.append(filename) 
        
        fnamesTest_label = []
        for filename in sorted(glob.glob(root + '/test_scaled_label_*.npz')):
            fnamesTest_label.append(filename) 

        fnamesTest_Vel = []
        for filename in sorted(glob.glob(root + '/test_scaled_vel_*.npz')):
            fnamesTest_Vel.append(filename) 

        fnamesTest_Env =[]
        for filename in sorted(glob.glob(root + '/test_scaled_env_*.npz')):
            fnamesTest_Env.append(filename) 

        fnamesTest_EE = []
        for filename in sorted(glob.glob(root + '/test_scaled_ee_*.npz')):
            fnamesTest_EE.append(filename) 

        fnamesTest_Foot = []
        for filename in sorted(glob.glob(root + '/test_scaled_fcontact_*.npz')):
            fnamesTest_Foot.append(filename) 

        return fnamesTest, fnamesTest_label, fnamesTest_Vel, fnamesTest_Env,fnamesTest_EE,fnamesTest_Foot    
        
    def __len__(self):
        return len(self.fnamesTest)

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.`
        The control is not masked
        """
        # # load data from batch index files
        # test_data = np.load(self.fnamesTest[idx])['clips'].astype(np.float32)
        # test_data_label = np.load(self.fnamesTest_label[idx])['clips'].astype(np.int)
        
        # # Joint positions
        # self.autoreg = test_data[:,:-self.condinfo] # info
        # self.cond = test_data[:,-self.condinfo:]
        # self.label = test_data_label[:]

        # load data from batch index files
        test_data = np.load(self.fnamesTest[idx])['clips'].astype(np.float32)
        test_data_vel = np.load(self.fnamesTest_Vel[idx])['clips'].astype(np.float32)
        test_data_label = np.load(self.fnamesTest_label[idx])['clips'].astype(np.int)
        test_data_env = np.load(self.fnamesTest_Env[idx])['clips'].astype(np.float32)
        test_data_ee = np.load(self.fnamesTest_EE[idx])['clips'].astype(np.float32)
        test_data_foot = np.load(self.fnamesTest_Foot[idx])['clips'].astype(np.float32)
        # Joint positions
        # self.autoreg = test_data[:,:-self.condinfo] # info
        # self.cond = test_data_vel[:,-self.condinfo:]
        # self.label = test_data_label[:]

        # sample = {'autoreg': self.autoreg[:,:], 'cond': self.cond, 'label':self.label}
        sample = {'autoreg': test_data, 'cond': test_data_vel,'label':test_data_label, 'env':test_data_env, 'ee': test_data_ee, 'foot':test_data_foot}    
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
