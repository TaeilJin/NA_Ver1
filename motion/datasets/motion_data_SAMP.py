import numpy as np
from torch.utils.data import Dataset
import glob
import os
from torch.utils.data import DataLoader
import torch

class TrainDataset_SAMP(Dataset):
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
        self.fnamesTrainX, self.fnamesTrainCond, self.fnamesTrainDescriptor, self.fnamesTrainLabel, self.fnamesTrainEnv = self._make_dataset(dataroot)
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
        self.zeromask_ee[-6:,1] = True # lower end effector 
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

        return fnamesTrainX, fnamesTrainCond, fnamesTrainDescriptor, fnamesTrainLabel, fnamesTrainEnv
                                                                                                                 
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

        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 0, 1)
        self.cond = np.swapaxes(self.cond, 0, 1)
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
        self.descriptor = np.swapaxes(self.descriptor,0,1)    
        # 202207
        self.env = np.swapaxes((self.env), 0, 1)

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
            
            # pose x 
            prev_x = self.x[:,:tt-10].copy()
            y_x = self.x[:,1:tt-9].copy()
                        
            # descriptor (sequence + env)
            prev_des = self.descriptor[:,:tt-10].copy()
            y_des = self.descriptor[:,1:tt-9].copy()
            
            # environment 
            y_env = self.env[:,1:tt-9].copy()

            # make previous and current state
            prev_state = np.concatenate((prev_x,prev_des),axis=0)
            y_state =np.concatenate((y_x,y_des),axis=0)
            
            # SAMP output
            sample = {'prev_state':prev_state,'env':y_env,'y_state':y_state}
            # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다.
            #sample = {'x': self.x[:,:], 'cond': cond_masked, 'ee_gt':self.ee_cond, 'ee_cond' :ee_cond_masked, 'descriptor' : des_masked, 'label':self.label[:,:]}
        else:
            sample = {'x': self.x[:,:], 'cond': self.cond[:,:],'ee_cond':self.ee_cond,'descriptor' : self.descriptor[:,:],'label':self.label[:,:]}
            
        return sample
class ValidDataset_SAMP(Dataset):
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
        # 20220419. label 을 넣었다  Valid
        self.fnamesValidX, self.fnamesValidCond, self.fnamesValidDescriptor, self.fnamesValidLabel, self.fnamesValidEnv = self._make_dataset(dataroot)
        
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
        self.zeromask_ee[-6:,1] = True # lower end effector 
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
        
        return fnamesValidX, fnamesValidCond, fnamesValidDescriptor, fnamesValidLabel, fnamesValidEnv                                                                     
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

        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 0, 1)
        self.cond = np.swapaxes(self.cond, 0, 1)
        # 20220419. descriptor 얻기 위해서 해당 condition 을 가져온다. Valid
        self.descriptor = np.swapaxes(self.descriptor, 0, 1)
        #
        self.env = np.swapaxes(self.env, 0, 1)

        # end effector
        self.ee_cond = np.zeros((self.ee_dim,self.x.shape[-1]),dtype=np.float32) 
        self.ee_cond[:3,:] = self.x[self.ee_HEAD_idx,:]
        self.ee_cond[(3):(3)+3,:] = self.x[self.ee_LH_idx,:]
        self.ee_cond[(6):(6)+3,:] = self.x[self.ee_RH_idx,:]
        self.ee_cond[(9):(9)+3,:] = self.x[self.ee_RF_idx,:]
        self.ee_cond[(12):(12)+3,:] = self.x[self.ee_LF_idx,:]

        if self.dropout>0.:
            n_feats, tt = self.x[:,:].shape
            cond_masked = self.cond[:,:].copy()
            
            # pose x 
            prev_x = self.x[:,:tt-10]
            y_x = self.x[:,1:tt-9]
                        
            # descriptor (sequence + env)
            prev_des = self.descriptor[:,:tt-10].copy()
            y_des = self.descriptor[:,1:tt-9].copy()
            
            # environment 
            y_env = self.env[:,1:tt-9]

            # make previous and current state
            prev_state = np.concatenate((prev_x,prev_des),axis=0)
            y_state =np.concatenate((y_x,y_des),axis=0)
            
            # SAMP output
            sample = {'prev_state':prev_state,'env':y_env,'y_state':y_state}
        else:
            sample = {'x': self.x[:,:], 'cond': self.cond[:,:],'ee_cond': self.ee_cond, 'descriptor':self.descriptor[:,:], 'label':self.label}
            
        return sample
class TestDataset_SAMP(Dataset):
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
        self.fnamesTest , self.fnamesTest_label, self.fnamesTest_Vel, self.fnamesTest_Env, self.fnamesTest_EE = self._make_dataset(dataroot)
    
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

        return fnamesTest, fnamesTest_label, fnamesTest_Vel, fnamesTest_Env,fnamesTest_EE    
        
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
        # Joint positions
        # self.autoreg = test_data[:,:-self.condinfo] # info
        # self.cond = test_data_vel[:,-self.condinfo:]
        # self.label = test_data_label[:]

        # sample = {'autoreg': self.autoreg[:,:], 'cond': self.cond, 'label':self.label}
        sample = {'autoreg': test_data, 'cond': test_data_vel,'label':test_data_label, 'env':test_data_env, 'ee': test_data_ee}    
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
    train_dataset = TrainDataset_SAMP(os.path.join(data_root,'mixamo_env_npz'),10, 0.7)
    data_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    # data_loader = get_dataloader('test', config)
    for batch in data_loader:
        # print_composite(batch)
        print(batch['x'])
