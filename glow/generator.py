import re
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from . import Experiment_utils

import sys
sys.path.append('holdens')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
import scipy.ndimage.filters as filters
import joblib
from scipy.spatial.transform import Rotation as R
import time

class Generator(object):
    def __init__(self, data, data_device, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # model relative
        self.data_device = data_device
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        self.data = data
        self.log_dir = log_dir
        #self.ee_cond = np.zeros((self.x.shape[0],self.ee_dim,self.x.shape[2]),dtype=np.float32) 
        self.ee_HEAD_idx = [(5)*3 +0,(5)*3 +1,(5)*3 +2]
        self.ee_LH_idx=[(9)*3 +0,(9)*3 +1,(9)*3 +2]
        self.ee_RH_idx=[(13)*3 +0,(13)*3 +1,(13)*3 +2]
        self.ee_RF_idx=[(17)*3 +0,(17)*3 +1,(17)*3 +2]
        self.ee_LF_idx=[(21)*3 +0,(21)*3 +1,(21)*3 +2]
        
        self.ee_dim = 5*3 # Head, LH, RH, RF, LF 순서로 가자
        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=hparams.Test.batch_size,
                                      num_workers=1,
                                      shuffle=False,
                                      drop_last=True)
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)
    
    def prepare_eecond(self, jt_data):
        # input data inside ee_cond 
        jt_data = np.swapaxes(jt_data,1,2)
        ee_cond = np.zeros((jt_data.shape[0],self.ee_dim,jt_data.shape[2]),dtype=np.float32) 
        ee_cond[:,:3,:] = jt_data[:,self.ee_HEAD_idx,:]
        ee_cond[:,(3):(3)+3,:] = jt_data[:,self.ee_LH_idx,:]
        ee_cond[:,(6):(6)+3,:] = jt_data[:,self.ee_RH_idx,:]
        #ee_cond[:,(9):(9)+3,:] = jt_data[:,self.ee_RF_idx,:]
        #ee_cond[:,(12):(12)+3,:] = jt_data[:,self.ee_LF_idx,:]
        ee_cond = torch.from_numpy(ee_cond)
        return ee_cond.to(self.data_device)
    
    def prepare_ee_upper(self, jt_data, head =False, hands=False):
        # input data inside ee_cond 
        jt_data = np.swapaxes(jt_data,1,2)
        ee_cond = np.ones((jt_data.shape[0],9,jt_data.shape[2]),dtype=np.float32) * 1e9
        if head == True:
            ee_cond[:,:3,:] = jt_data[:,self.ee_HEAD_idx,:]
        if hands == True:
            ee_cond[:,(3):(3)+3,:] = jt_data[:,self.ee_LH_idx,:]
            ee_cond[:,(6):(6)+3,:] = jt_data[:,self.ee_RH_idx,:]
            
        ee_cond = torch.from_numpy(ee_cond)
        return ee_cond.to(self.data_device)

    def prepare_Rot_eecond(self,ee_data):
        ee = ee_data.copy()
        ee[:,:,:9] =0 # position is 0
        #ee[:,:,9:15] = 0 # lower position is 0

        ee[:,:,15:24] = 0 # rotation is 0
        #ee[:,:,24:] = 0 # Lower rotation is 0
        
        ee_cond = torch.from_numpy(np.swapaxes(ee,1,2))
        return ee_cond.to(self.data_device)

    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)
    
    def generate_0515_sample_withRef(self, demo_file_path, scaler_path, graph, graph_cond, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        
        """ generate data """
        filename = demo_file_path
        scaler = joblib.load(f'{scaler_path}/mixamo.pkl')

        clips = np.loadtxt(filename, delimiter=" ")
            
        datas = clips.copy().astype(np.float32)

        """ position """
        positions = datas[:,:66]
        """ rotation """
        rotations = datas[:,66:132]
        """ end-effector position """
        ee_positions = datas[:,132:132+5*3]
        """ end-effector rotation """
        ee_rotations = datas[:,(132+5*3):(132+5*3*2)]
        """ environment """
        env_dim = 2640
        env_centerpos = datas[:,-(env_dim*4+ (3*2)):-(env_dim+(3*2))]
        env_occupancy = datas[:,-(env_dim + (3*2)):-(3*2)]
        """ root velocity """
        root_positions = datas[:,-(3*2):-(3)]
        root_forwards = datas[:,-(3):]
        # linear
        velocity = (root_positions[1:,:] - root_positions[:-1,:]).copy()
        # rotation
        target = np.array([[0,0,1]]).repeat(len(root_forwards), axis=0)
        rotation = Quaternions.between(root_forwards, target).copy()#[:,np.newaxis]    
        #calc
        velocity = rotation[1:] * velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
        #first velocity is zero 
        zero_vel = np.array([[0,0,0]])
        velocity = np.concatenate([zero_vel,velocity], axis=0)
        rvelocity = np.concatenate([zero_vel[:,0],rvelocity],axis=0)[:,np.newaxis] 
        """ gen testdata"""
        scaling_data = np.concatenate((positions,ee_positions,ee_rotations,velocity[:,0:1],velocity[:,2:3],rvelocity),axis=-1)
        scaling_data = scaler.transform(scaling_data).astype(np.float32)
        
        autoreg_all = scaling_data[np.newaxis,:,:66]
        control_all = scaling_data[np.newaxis,:,-3:]
        env_all = env_occupancy[np.newaxis,:,:]
        ee_all = scaling_data[np.newaxis,:,66:66+30]
        #
        with torch.no_grad():
            batch = self.test_batch

            # autoreg_all = batch["autoreg"].cpu().numpy()
            # control_all = batch["cond"].cpu().numpy()
            
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            ee_all = np.zeros((nn,n_timesteps,30))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            sampled_z_all = graph.loss_fn_GMM.sample_all(nn)
            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)

            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                ee_cond = self.prepare_eecond(refpose.copy())
                #ee_cond = torch.zeros(nn,15,1)

                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                prob = prob.unsqueeze(1)

                # sample from Moglow
                sampled_z_label = torch.bmm(prob,sampled_z_all).permute(0,2,1).clone().detach()

                sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        #self.data.save_animation_withRef(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.scaler = scaler
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))


    def gen_world_pos_data(self,i_joints,i_rootvel):
        joints = i_joints.copy()
        roots = i_rootvel.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((joints.shape[0],3))
        
        root_dx, root_dz, root_dr = roots[...,-3], roots[...,-2], roots[...,-1]
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
            rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot

            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = joints[i,:,0] + translation[0,0]
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
                       
            
            translations[i,:] = translation
        
        return joints, translations

    def generate_txt_file(self,filepath,name,val,joints_world,trajectory_world):
        nTimesteps, nJoints, nFeats = joints_world.shape
        nTimesteps, nRootpos = trajectory_world.shape
        
        joints_w = joints_world.reshape(nTimesteps,-1)

        dirname = os.path.dirname(filepath)
        file_path = os.path.splitext(filepath)[0]
        file_name = file_path.split('/')[-1]

        np.savetxt(f"{dirname}/{file_name}_{name}_joints_{val}.csv",joints_w,delimiter=",")
        np.savetxt(f"{dirname}/{file_name}_{name}_trajectory_{val}.txt",trajectory_world,delimiter=" ")


    def generate_sample_withRef(self, graph, eps_std=1.0, step=0, counter=0, autoreg_all=None,control_all=None,env_all=None):
        print("generate_sample")
        graph.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            if autoreg_all.any() == None:
                autoreg_all = batch["autoreg"].cpu().numpy()
                control_all = batch["cond"].cpu().numpy()
                env_all = batch["env"].cpu().numpy()
            
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            ee_all = np.zeros((nn,n_timesteps,30))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            sampled_z_random = graph.distribution.sample((nn,66,1), eps_std, device=self.data_device)
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                env = env_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                ee_cond = self.prepare_eecond(refpose.copy())
                # prepare conditioning for moglow (control + previous poses)
                cond = self.prepare_cond(autoreg.copy(), control.copy())
                
                env = torch.from_numpy(env).permute(0,2,1).to(self.data_device)
                cond_env = torch.cat((cond,env),dim=1).to(self.data_device)

                #no ee condition
                #ee_cond = None
                #cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                #cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                
                #cond = self.prepare_cond(autoreg.copy(), control.copy())
                # sample from Moglow
                sampled_z_label = sampled_z_random.clone().detach()

                sampled = graph(z=sampled_z_label, cond=cond_env, ee_cond = ee_cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        #self.data.save_animation_withRef(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        
        joints_all_unscaled = Experiment_utils.unNormalize_motion(sampled_all, self.data.scaler) 
        vel_all_unscaled = Experiment_utils.unNormalize_vel(control_all, self.data.scaler)
        ref_joints_all_unscaled = Experiment_utils.unNormalize_motion(reference_all, self.data.scaler) 
        
        joints, trajectory = self.gen_world_pos_data(joints_all_unscaled[0], vel_all_unscaled[0])
       
        self.generate_txt_file(os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'), "sample", 0, joints, trajectory)


    def generate_motion_woGMM(self, graph, graph_cond, control_seq, seqlen, z,
    env_all=None,test_gt_all=None,ee_erase_idx=None):
        nn, n_timesteps, n_feats = control_seq.shape 
        # sequence information length
        seqlen = self.seqlen
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()
        
        env_all = torch.from_numpy(env_all).to(self.data_device)
        # z = (B,1,Fpose)
        sampled_all = np.zeros((nn, n_timesteps, z.shape[-1]))
        autoreg = np.zeros((nn, seqlen, z.shape[-1]), dtype=np.float32) #initialize from a mean pose
       
        sampled_all[:,:seqlen,:] = autoreg 
        ee_cond = torch.zeros((nn,15,1)) # ee_cond is zero
        # 
        # Loop through control sequence and generate new data
        for i in range(0,n_timesteps-seqlen):
            control = control_seq[:,i:(i+seqlen+1),:]
            
            if test_gt_all is not None:
                refpose = test_gt_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                jt_data = np.swapaxes(refpose,1,2).copy()
                jt_data = torch.from_numpy(jt_data).to(self.data_device)

                ee_cond[:,:3,:] = jt_data[:,self.ee_HEAD_idx,:]
                ee_cond[:,(3):(3)+3,:] = jt_data[:,self.ee_LH_idx,:]
                ee_cond[:,(6):(6)+3,:] = jt_data[:,self.ee_RH_idx,:]
                ee_cond[:,(9):(9)+3,:] = jt_data[:,self.ee_RF_idx,:]
                ee_cond[:,(12):(12)+3,:] = jt_data[:,self.ee_LF_idx,:]
                
                ee_cond[:,ee_erase_idx,:] = 0 
                #ee_cond = torch.from_numpy(ee_cond).to(self.data_device)
                
                #ee_cond = self.prepare_eecond(refpose.copy())
            
            # prepare conditioning for moglow (control + previous poses)
            descriptor = self.prepare_cond(autoreg.copy(), control.copy())
            env = env_all[:,(i+seqlen):(i+seqlen+1),:].permute(0,2,1)
            descriptor_env = torch.cat((descriptor,env),dim=1)

            # current condition
            cond = control_seq[:,(i+seqlen):(i+seqlen+1),:]
            cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
            cond_env = torch.cat((cond,env),dim=1).to(self.data_device)

            # descriptor encoder 
            nBatch, nFeatures, nTimesteps = descriptor_env.shape
            des_cond = descriptor_env.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
            out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
            prob = out_net['prob_cat']
            #prob = prob.unsqueeze(1)
            # sample from Moglow z = weighted_sum(mu_c * z + mu_var)
            sampled_z_label = z.permute(0,2,1).clone().detach().to(self.data_device)
            prob = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach().to(self.data_device)
            # sample current pose
            sampled = graph(z=sampled_z_label, cond=cond_env, ee_cond = ee_cond, label=prob, eps_std=1.0, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled # sampled
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
        
        return sampled_all

    def generate_motion(self, graph, graph_cond, control_seq, seqlen, z,
    env_all=None,test_gt_all=None,ee_erase_idx=None):
        nn, n_timesteps, n_feats = control_seq.shape 
        # sequence information length
        seqlen = self.seqlen
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()
        
        env_all = torch.from_numpy(env_all).to(self.data_device)
        # z = (B,1,Fpose)
        sampled_all = np.zeros((nn, n_timesteps, z.shape[-1]))
        autoreg = np.zeros((nn, seqlen, z.shape[-1]), dtype=np.float32) #initialize from a mean pose
       
        sampled_all[:,:seqlen,:] = autoreg 
        ee_cond = torch.zeros((nn,15,1)) # ee_cond is zero
        # 
        # Loop through control sequence and generate new data
        for i in range(0,n_timesteps-seqlen):
            control = control_seq[:,i:(i+seqlen+1),:]
            
            if test_gt_all is not None:
                refpose = test_gt_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                jt_data = np.swapaxes(refpose,1,2).copy()
                jt_data = torch.from_numpy(jt_data).to(self.data_device)

                ee_cond[:,:3,:] = jt_data[:,self.ee_HEAD_idx,:]
                ee_cond[:,(3):(3)+3,:] = jt_data[:,self.ee_LH_idx,:]
                ee_cond[:,(6):(6)+3,:] = jt_data[:,self.ee_RH_idx,:]
                ee_cond[:,(9):(9)+3,:] = jt_data[:,self.ee_RF_idx,:]
                ee_cond[:,(12):(12)+3,:] = jt_data[:,self.ee_LF_idx,:]
                
                ee_cond[:,ee_erase_idx,:] = 0 
                

                #ee_cond = torch.from_numpy(ee_cond).to(self.data_device)
                #ee_cond = self.prepare_eecond(refpose.copy())
            
            #start = time.time()  # 시작 시간 저장

            # prepare conditioning for moglow (control + previous poses)
            descriptor = self.prepare_cond(autoreg.copy(), control.copy())
            env = env_all[:,(i+seqlen):(i+seqlen+1),:].permute(0,2,1)
            descriptor_env = torch.cat((descriptor,env),dim=1)

            # current condition
            cond = control_seq[:,(i+seqlen):(i+seqlen+1),:]
            cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
            cond_env = torch.cat((cond,env),dim=1).to(self.data_device)

            # descriptor encoder 
            nBatch, nFeatures, nTimesteps = descriptor_env.shape
            des_cond = descriptor_env.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
            out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
            prob = out_net['prob_cat']
            prob = prob.unsqueeze(1)

            #print("descriptor network time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

            # sample from Moglow z = weighted_sum(mu_c * z + mu_var)
            sampled_z_label = torch.bmm(prob,z).permute(0,2,1).clone().detach()
            # sample current pose
            sampled = graph(z=sampled_z_label, cond=cond_env, ee_cond = ee_cond, eps_std=1.0, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            #print("generator network time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled # sampled
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
        
        return sampled_all
    
    def generate_motion_label(self, graph, graph_cond, control_seq, seqlen, z,
    env_all=None,test_gt_all=None,ee_erase_idx=None):
        nn, n_timesteps, n_feats = control_seq.shape 
        # sequence information length
        seqlen = self.seqlen
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()
        
        env_all = torch.from_numpy(env_all).to(self.data_device)
        # z = (B,1,Fpose)
        sampled_all = np.zeros((nn, n_timesteps, z.shape[-1]))
        autoreg = np.zeros((nn, seqlen, z.shape[-1]), dtype=np.float32) #initialize from a mean pose
       
        sampled_all[:,:seqlen,:] = autoreg 
        ee_cond = torch.zeros((nn,15,1)) # ee_cond is zero
        # 
        # Loop through control sequence and generate new data
        for i in range(0,n_timesteps-seqlen):
            control = control_seq[:,i:(i+seqlen+1),:]
            
            if test_gt_all is not None:
                refpose = test_gt_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                jt_data = np.swapaxes(refpose,1,2).copy()
                jt_data = torch.from_numpy(jt_data).to(self.data_device)

                ee_cond[:,:3,:] = jt_data[:,self.ee_HEAD_idx,:]
                ee_cond[:,(3):(3)+3,:] = jt_data[:,self.ee_LH_idx,:]
                ee_cond[:,(6):(6)+3,:] = jt_data[:,self.ee_RH_idx,:]
                ee_cond[:,(9):(9)+3,:] = jt_data[:,self.ee_RF_idx,:]
                ee_cond[:,(12):(12)+3,:] = jt_data[:,self.ee_LF_idx,:]
                
                ee_cond[:,ee_erase_idx,:] = 0 
                #ee_cond = torch.from_numpy(ee_cond).to(self.data_device)
                
                #ee_cond = self.prepare_eecond(refpose.copy())
            
            # prepare conditioning for moglow (control + previous poses)
            descriptor = self.prepare_cond(autoreg.copy(), control.copy())
            env = env_all[:,(i+seqlen):(i+seqlen+1),:].permute(0,2,1)
            descriptor_env = torch.cat((descriptor,env),dim=1)

            # current condition
            cond = control_seq[:,(i+seqlen):(i+seqlen+1),:]
            cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
            cond_env = torch.cat((cond,env),dim=1).to(self.data_device)

            # descriptor encoder 
            nBatch, nFeatures, nTimesteps = descriptor_env.shape
            des_cond = descriptor_env.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
            out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
            prob = out_net['prob_cat']
            prob_z = prob.unsqueeze(1).clone().detach()

            #print("descriptor network time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

            # sample from Moglow z = weighted_sum(mu_c * z + mu_var)
            sampled_z_label = torch.bmm(prob_z,z).permute(0,2,1).clone().detach().to(self.data_device)
            
            prob_label = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach().to(self.data_device)
            # sample current pose
            sampled = graph(z=sampled_z_label, cond=cond_env, ee_cond = ee_cond, label=prob_label, eps_std=1.0, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled # sampled
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
        
        return sampled_all


    def generate_diverse_motion_withDlow(self,demo_file_path, scaler_path, graph, graph_gen, graph_cond, eps_std=1.0, step=0):
        with torch.no_grad():
            # 
            graph.eval()
            graph_gen.eval()
            graph_cond.eval()
            env_dim = 2640
            # #
            # control_all = self.test_batch["cond"].cpu().numpy()
            # gt_all = self.test_batch["autoreg"].cpu().numpy()
            # env_all = self.test_batch["env"].cpu().numpy()
            # #
            # nFeats = gt_all.shape[-1]
            # seqlen = self.seqlen
            # #
            # control_all = control_all.reshape(-1,3)
            # control_all = np.expand_dims(control_all,axis=0).copy()

            # gt_all = gt_all.reshape(-1,nFeats)
            # gt_all = np.expand_dims(gt_all,axis=0).copy()

            # env_all = env_all.reshape(-1,env_dim)
            # env_all = np.expand_dims(env_all,axis=0).copy()
            """ generate data """
            filename = demo_file_path
            scaler = joblib.load(f'{scaler_path}/mixamo.pkl')

            clips = np.loadtxt(filename, delimiter=" ")
                
            datas = clips.copy().astype(np.float32)

            """ position """
            positions = datas[:,:66]
            """ rotation """
            rotations = datas[:,66:132]
            """ end-effector position """
            ee_positions = datas[:,132:132+5*3]
            """ end-effector rotation """
            ee_rotations = datas[:,(132+5*3):(132+5*3*2)]
            """ environment """
            env_dim = 2640
            env_centerpos = datas[:,-(env_dim*4+ (3*2)):-(env_dim+(3*2))]
            env_occupancy = datas[:,-(env_dim + (3*2)):-(3*2)]
            """ root velocity """
            root_positions = datas[:,-(3*2):-(3)]
            root_forwards = datas[:,-(3):]
            # linear
            velocity = (root_positions[1:,:] - root_positions[:-1,:]).copy()
            # rotation
            target = np.array([[0,0,1]]).repeat(len(root_forwards), axis=0)
            rotation = Quaternions.between(root_forwards, target).copy()#[:,np.newaxis]    
            #calc
            velocity = rotation[1:] * velocity
            rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
            #first velocity is zero 
            zero_vel = np.array([[0,0,0]])
            velocity = np.concatenate([zero_vel,velocity], axis=0)
            rvelocity = np.concatenate([zero_vel[:,0],rvelocity],axis=0)[:,np.newaxis] 
            """ gen testdata"""
            scaling_data = np.concatenate((positions,ee_positions,ee_rotations,velocity[:,0:1],velocity[:,2:3],rvelocity),axis=-1)
            scaling_data = scaler.transform(scaling_data).astype(np.float32)
            
            gt_all = scaling_data[np.newaxis,:,:66]
            control_all = scaling_data[np.newaxis,:,-3:]
            env_all = env_occupancy[np.newaxis,:,:]
            ee_all = scaling_data[np.newaxis,:,66:66+30]


            #
            seqlen = self.seqlen
            nstart = -181 #1280
            nend = -1 #nstart + 180
            control_all = control_all[:,nstart:nend,:]
            gt_all = gt_all[:,nstart:nend,:]
            env_all = env_all[:,nstart:nend,:]
            ee_all = ee_all[:,nstart:nend,:]
            # z from multi gaussians
            nFrames = control_all.shape[1]
            
            #
            # control -> generate Z 
            num_clips = graph.nSample
            nTimesteps = 1
            cond = torch.from_numpy(control_all).permute(0,2,1).to(self.data_device) # (nBatch,nFcond, nTimesteps)
            env = torch.from_numpy(env_all).permute(0,2,1).to(self.data_device)
            cond_env = torch.cat((cond,env),dim=1).to(self.data_device)
            graph.init_lstm_hidden()
            gen_z_value, a, b = graph(cond_env) # (B,Fpose*num_data)
            nBatch, nFeats_nData = gen_z_value.shape
            nFpose = nFeats_nData // num_clips
            gen_z_value = gen_z_value.reshape(nBatch, num_clips,-1) #(B, numData, Fpose)

            nD_sampled = np.zeros((nBatch,num_clips,nFrames,nFpose))
            
            self.data.scaler = scaler
                
            for nD in range(num_clips):
                gen_z_eps = gen_z_value[:,nD,:] #(B,Fpose)

                # input z epsilion value
                gen_z_eps = gen_z_eps.repeat_interleave(nTimesteps, dim=0).unsqueeze(1) #(BxTimesteps,Fpose)
                gen_z_eps = gen_z_eps.repeat_interleave(graph_gen.gaussian_size,dim=1) #(BxTimesteps,classess,Fpose)
                
                # means, vars
                means_nD = graph_gen.means.clone().detach() #(classess,Fpose)
                vars_nD = graph_gen.vars.clone().detach() 
                means_nD = means_nD.unsqueeze(0).repeat_interleave(nBatch*nTimesteps,dim=0).to(self.data_device) #(BxTimesteps,classess,Fpose)
                vars_nD = vars_nD.unsqueeze(0).repeat_interleave(nBatch*nTimesteps,dim=0).to(self.data_device) #(BxTimesteps,classess,Fpose)
                # calc input
                gen_z_eps = means_nD + gen_z_eps * vars_nD #(BxTimesteps,classess,Fpose)
                # generate motion
                sampled = self.generate_motion(graph_gen,graph_cond,control_all,seqlen,gen_z_eps,env_all, None)
                nD_sampled[:,nD,:,:] = sampled

                self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled, gt_all, os.path.join(self.log_dir, f'Sample_{nD}_seq{seqlen}_frame{nstart}_{nend-nstart+1}_{str(step)}k'))

                #self.data.save_animation(control_all, sampled,os.path.join(self.log_dir, f'Sample_{nD}_seq{seqlen}_frame{nstart}_{nend-nstart+1}_{str(step)}k'))
                #self.data.save_animation_withRef(control_all, sampled, gt_all, os.path.join(self.log_dir, f'Sample_{nD}_seq{seqlen}_frame{nstart}_{nend-nstart+1}_{str(step)}k'))
            # calc Score (K,nBatch,nTimesteps,nFeats)
            apd_score = Experiment_utils.get_APD_Score(np.concatenate((ee_all,control_all),axis=-1),nD_sampled.swapaxes(0,1),num_clips,self.data.scaler)

            # print
            print(f'see_{apd_score}')

    def generate_ROT_0515_sample_withRef(self, demo_file_path, scaler_path, graph, graph_cond, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        
        """ generate data """
        filename = demo_file_path
        scaler = joblib.load(f'{scaler_path}/mixamo.pkl')

        clips = np.loadtxt(filename, delimiter=" ")
            
        datas = clips.copy().astype(np.float32)

        """ position """
        positions = datas[:,:66]
        """ rotation """
        rotations = datas[:,66:132]
        """ end-effector position """
        ee_positions = datas[:,132:132+5*3]
        """ end-effector rotation """
        ee_rotations = datas[:,(132+5*3):(132+5*3*2)]
        """ environment """
        env_dim = 2640
        env_centerpos = datas[:,-(env_dim*4+ (3*2)):-(env_dim+(3*2))]
        env_occupancy = datas[:,-(env_dim + (3*2)):-(3*2)]
        """ root velocity """
        root_positions = datas[:,-(3*2):-(3)]
        root_forwards = datas[:,-(3):]
        # linear
        velocity = (root_positions[1:,:] - root_positions[:-1,:]).copy()
        # rotation
        target = np.array([[0,0,1]]).repeat(len(root_forwards), axis=0)
        rotation = Quaternions.between(root_forwards, target).copy()#[:,np.newaxis]    
        #calc
        velocity = rotation[1:] * velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
        #first velocity is zero 
        zero_vel = np.array([[0,0,0]])
        velocity = np.concatenate([zero_vel,velocity], axis=0)
        rvelocity = np.concatenate([zero_vel[:,0],rvelocity],axis=0)[:,np.newaxis] 
        """ gen testdata"""
        scaling_data = np.concatenate((positions[:,:3],rotations,ee_positions,ee_rotations,velocity[:,0:1],velocity[:,2:3],rvelocity),axis=-1)
        scaling_data = scaler.transform(scaling_data).astype(np.float32)
        
        autoreg_all = scaling_data[np.newaxis,:,:69]
        control_all = scaling_data[np.newaxis,:,-3:]
        env_all = env_occupancy[np.newaxis,:,:]
        ee_all = scaling_data[np.newaxis,:,69:69+30]
        #
        with torch.no_grad():
            batch = self.test_batch

            # autoreg_all = batch["autoreg"].cpu().numpy()
            # control_all = batch["cond"].cpu().numpy()
            
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            sampled_z_all = graph.loss_fn_GMM.sample_all(nn)
            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                #ee_cond = self.prepare_eecond(refpose.copy())
                ee_cond = ee_all[:,(i+seqlen):(i+seqlen+1),:]
                ee_cond = self.prepare_Rot_eecond(ee_cond)

                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                prob = prob.unsqueeze(1)

                # sample from Moglow
                sampled_z_label = torch.bmm(prob,sampled_z_all).permute(0,2,1).clone().detach()

                sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        #self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))

   
    def generate_ROT_sample_withRef_cond(self, graph, graph_cond, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            sampled_z_all = graph.loss_fn_GMM.sample_all(nn)
            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                #ee_cond = self.prepare_eecond(refpose.copy())
                ee_cond = ee_all[:,(i+seqlen):(i+seqlen+1),:]
                ee_cond = self.prepare_Rot_eecond(ee_cond)

                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                prob = prob.unsqueeze(1)

                # sample from Moglow
                sampled_z_label = torch.bmm(prob,sampled_z_all).permute(0,2,1).clone().detach()

                sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        #self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))
     

    def generate_sample_withRef_cond_foot(self, graph, graph_cond, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            foot_all = batch["foot"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            if hasattr(graph, "module"):
                sampled_z_all = graph.module.loss_fn_GMM.sample_all(nn)
            else:
                sampled_z_all = graph.loss_fn_GMM.sample_all(nn)

            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)
               
                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                
                prob_zvalue = prob.unsqueeze(1).clone().detach()
                prob_label = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
                # sample from Moglow
                sampled_z_label = torch.bmm(prob_zvalue,sampled_z_all).permute(0,2,1).clone().detach()

                
                # foot velocity
                foot = foot_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                foot = torch.from_numpy(np.swapaxes(foot,1,2)).to(self.data_device)


                


                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                ee_cond = None
                if upper_cond is None:
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob_label,foot=foot, eps_std=1.0, reverse=True)
                else:
                    upper_cond = torch.from_numpy(np.swapaxes(refpose,1,2)).to(sampled_z_label.device)
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob_label, foot=foot, eps_std=1.0, reverse=True, upper_cond = upper_cond[:,:42,:])
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))

    def generate_sample_withRef_foot_estimator(self, graph, graph_cond, graph_foot, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        graph_foot.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            foot_all = batch["foot"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            if hasattr(graph, "module"):
                sampled_z_all = graph.module.loss_fn_GMM.sample_all(nn)
            else:
                sampled_z_all = graph.loss_fn_GMM.sample_all(nn)

            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)
               
                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                
                prob_zvalue = prob.unsqueeze(1).clone().detach()
                prob_label = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
                # sample from Moglow
                sampled_z_label = torch.bmm(prob_zvalue,sampled_z_all).permute(0,2,1).clone().detach()

                #start = time.time()
                
                # foot velocity
                scene_feature = torch.cat((descriptor,env),dim=1)
                foot_all = graph_foot(cond=scene_feature, phi = prob.clone().detach())
                # B*T,F -> B,T,F -> B,F,T
                foot_all_l = graph_foot.m_sig(foot_all[:,:2])
                foot_all_r = graph_foot.m_sig(foot_all[:,2:])
                
                foot_all_l[foot_all_l >0.5] = 1
                foot_all_l[foot_all_l<0.5] = 0

                foot_all_r[foot_all_r >0.5] = 1
                foot_all_r[foot_all_r<0.5] = 0
                

                foot_all_l = foot_all_l.unsqueeze(1).clone().detach() # (B*T,1,F)
                foot_all_l = foot_all_l.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
                foot_all_r = foot_all_r.unsqueeze(1).clone().detach() # (B*T,1,F)
                foot_all_r = foot_all_r.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
                foot = torch.cat((foot_all_l,foot_all_r),dim=1)

                #start = time.time()
                if upper_cond is None:
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob_label,foot=foot, eps_std=1.0, reverse=True)
                else:
                    upper_cond = torch.from_numpy(np.swapaxes(refpose,1,2)).to(sampled_z_label.device)
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob_label, foot=foot, eps_std=1.0, reverse=True, upper_cond = upper_cond[:,:42,:])
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))


    def generate_sample_withRef_History_foot(self, graph, graph_cond, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            foot_all = batch["foot"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            if hasattr(graph, "module"):
                sampled_z_all = graph.module.loss_fn_GMM.sample_all(nn)
            else:
                sampled_z_all = graph.loss_fn_GMM.sample_all(nn)

            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
               
                
                prob_zvalue = prob.unsqueeze(1).clone().detach()
                prob_label = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
                # sample from Moglow
                sampled_z_label = torch.bmm(prob_zvalue,sampled_z_all).permute(0,2,1).clone().detach()


                # condition (descriptor + env)
                #cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                #cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                #cond = torch.cat((cond,env),dim=1)
                cond = torch.cat((descriptor,env),dim=1)
                
                # foot velocity
                foot = foot_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                foot = torch.from_numpy(np.swapaxes(foot,1,2)).to(self.data_device)


                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                if upper_cond is None:
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob_label, foot = foot, eps_std=1.0, reverse=True)
                else:
                    upper_cond = torch.from_numpy(np.swapaxes(refpose,1,2)).to(sampled_z_label.device)
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob_label, foot = foot, eps_std=1.0, reverse=True, upper_cond = upper_cond[:,:42,:])
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        
    def generate_sample_withRef_cond_moglow(self, graph, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                descriptor = torch.cat((descriptor,env),dim=1)
     
                # sample from Moglow
                sampled_z_label = torch.zeros((nBatch,66,1)).to(self.data_device)

                sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, eps_std=1.0, reverse=True)

                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))
    

    def generate_sample_withRef_cond_woGMM(self, graph, graph_cond, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            # sample from Moglow
            sampled_z_label = torch.zeros((nBatch,66,1)).to(self.data_device)


            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                
                prob = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
               
                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                if upper_cond is None:
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob, eps_std=1.0, reverse=True)
                else:
                    upper_cond = torch.from_numpy(np.swapaxes(refpose,1,2)).to(sampled_z_label.device)
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob, eps_std=1.0, reverse=True, upper_cond = upper_cond[:,:42,:])
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))
    
    def generate_sample_withRef_History_woGMM(self, graph, graph_cond, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            if hasattr(graph, "module"):
                sampled_z_all = graph.module.loss_fn_GMM.sample_all(nn)
            else:
                sampled_z_all = graph.loss_fn_GMM.sample_all(nn)

            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                
                prob = prob.reshape(nBatch,nTimesteps,-1).permute(0,2,1).clone().detach()
                
                # sample from Moglow
                sampled_z_label = torch.zeros((nBatch,66,1)).to(self.data_device)

                # condition (descriptor + env)
                #cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                #cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                #cond = torch.cat((cond,env),dim=1)
                cond = torch.cat((descriptor,env),dim=1)
                

                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                if upper_cond is None:
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob, eps_std=1.0, reverse=True)
                else:
                    upper_cond = torch.from_numpy(np.swapaxes(refpose,1,2)).to(sampled_z_label.device)
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,label = prob, eps_std=1.0, reverse=True, upper_cond = upper_cond[:,:42,:])
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))


    def generate_sample_withRef_MCFlow(self, graph, graph_cond, graph_im, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        graph_im.eval()

        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                graph_im.init_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            if hasattr(graph, "module"):
                sampled_z_all = graph.module.loss_fn_GMM.sample_all(nn)
            else:
                sampled_z_all = graph.loss_fn_GMM.sample_all(nn)

            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                prob = prob.unsqueeze(1)


                # generate mask 
                jt_data = torch.from_numpy(np.swapaxes(refpose.copy(),1,2)).to(self.data_device)

                masked_init = torch.zeros(jt_data.shape).to(jt_data.device)
                masked_init = graph.flow.select_layer_u.addEndEffectorElement(masked_init,ee_cond)
                masked_init = masked_init.bool() 
                jt_data = jt_data * masked_init
                
                # x -> z
                z, _ = graph(x=jt_data, cond=cond, ee_cond = ee_cond)             
                # imputation
                z_hat = graph_im(z.permute(0, 2, 1)).permute(0, 2, 1)
                # hat to inverse
                sampled = graph(z= z_hat, cond=cond, ee_cond=ee_cond, eps_std=1.0, reverse=True)
                                
                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))

    def generate_sample_withRef_cond(self, graph, graph_cond, upper_cond =None, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        graph_cond.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            
            if hasattr(graph, "module"):
                sampled_z_all = graph.module.loss_fn_GMM.sample_all(nn)
            else:
                sampled_z_all = graph.loss_fn_GMM.sample_all(nn)

            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nn,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                # 전체 포즈에서 end-effector condition 을 만들어야한다
                # ee_cond = self.prepare_ee_upper(refpose.copy(),head=True)
                # #ee_cond = torch.zeros(nn,15,1)
                # x_head = ee_cond[:,:3,:]
                # x_hand = ee_cond[:,3:,:]
                # if torch.all(x_head > 1e8):
                #     x_head = None
                # if torch.all(x_hand > 1e8):
                #     x_hand = None
                ee_cond =self.prepare_eecond(refpose.copy())


                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)

                # condition (vel + env)
                cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)
                cond = torch.cat((cond,env),dim=1)

                # descriptor (sequence + env)
                nBatch, nFeatures, nTimesteps = descriptor.shape
                des_cond = descriptor.permute(0,2,1).reshape(-1,nFeatures).clone().detach()
                des_env = env.permute(0,2,1).reshape(nBatch*nTimesteps,-1).clone().detach()
                des_cond = torch.cat((des_cond,des_env),dim=-1)
                
                out_net = graph_cond.network(des_cond,0.7,graph_cond.hard_gumbel)
                prob = out_net['prob_cat']
                prob = prob.unsqueeze(1)

                # sample from Moglow
                sampled_z_label = torch.bmm(prob,sampled_z_all).permute(0,2,1).clone().detach()

                #start = time.time()
                # graph.to('cpu')
                # sampled_z_label = sampled_z_label.to('cpu')
                # cond = cond.to('cpu')
                # ee_cond = ee_cond.to('cpu')
                if upper_cond is None:
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,eps_std=1.0, reverse=True)
                else:
                    upper_cond = torch.from_numpy(np.swapaxes(refpose,1,2)).to(sampled_z_label.device)
                    sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond,eps_std=1.0, reverse=True, upper_cond = upper_cond[:,:42,:])
                
                #sampled = graph(z=sampled_z_label, cond=cond, ee_cond = ee_cond, x_head = x_head, x_hand = x_hand, eps_std=eps_std, reverse=True)
                
                #print(" computation time: " , time.time()-start)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        #self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))
        
    def generate_sample_withRef_SAMP(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph.eval()
        
        with torch.no_grad():
            batch = self.test_batch

            autoreg_all = batch["autoreg"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()
            env_all = batch["env"].cpu().numpy()
            ee_all = batch["ee"].cpu().numpy()
            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead
            
            #
            nn,n_timesteps,n_feats = autoreg_all.shape
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg
            
            sampled_z_all = torch.normal(mean=torch.zeros((nn,32)),
                           std=torch.ones((nn,32)) * eps_std)
            sampled_z_all = sampled_z_all.to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen):
                control = control_all[:,i:(i+seqlen+1),:]
                refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
                
                # prepare conditioning for moglow (control + previous poses)
                descriptor = self.prepare_cond(autoreg.copy(), control.copy())
                env = env_all[:,(i+seqlen):(i+seqlen+1),:]
                env = torch.from_numpy(np.swapaxes(env,1,2)).to(self.data_device)
                
                #
                init_pose = torch.from_numpy(refpose).to(self.data_device).permute(0,2,1)
                p_prev = torch.cat((init_pose,descriptor),dim=1)[:,:,0] # 693 으로
                I = env[:,:,0]

                # sample from Moglow
                sampled_z_label = sampled_z_all.clone().detach()

                if i !=0:
                    p_prev[:,:66] = pred_state[:,:66]
                    
                pred_state = graph.decoder(sampled_z_label, p_prev, I)
                sampled = pred_state.detach().clone().cpu().numpy()

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled[:,:66] # sampled
                reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:66]), axis=1)
                
        # store the generated animations
        self.data.save_animation_withRef(np.concatenate((ee_all,control_all),axis=-1), sampled_all, reference_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_UnityFile(sampled_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_T1_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_animation_UnityFile(reference_all, ee_all, control_all, os.path.join(self.log_dir, f'Env_REF'))
    
    def generate_test_label(self,graph,graph_cond,seqlen,test_gt,test_cond,test_env,N=10):

        print("generate_sample")
        graph.eval()
        graph_cond.eval()
    
        # generate N = 10 samples
        # test_gt = test_gt[0:1,...]
        # test_cond = test_cond[0:1,...]
        # test_env = test_env[0:1,...]
        nBatch, nTimesteps, nFeatures = test_gt.shape
        Total_samples = np.zeros((N,nBatch,nTimesteps,nFeatures)) # (N, B, T,F)
        Total_samples_forGT = np.zeros((2,nBatch,nTimesteps,nFeatures)) # (N, B, T,F)
        Total_samples_forGT[0] = test_gt

        # evaluate GT result
        #Total_samples[0] = test_gt
        #apd_score = Experiment_utils.get_motion_APD_Score(Total_samples[0:2],totalClips=2,scaler=self.data.scaler)
        
        # save animation
        #test_gt_unNorm = Experiment_utils.unNormalize_motion(test_gt, self.data.scaler)
        #test_init_unNorm = Experiment_utils.unNormalize_motion(Total_samples[1], self.data.scaler)
        #test_cond_unNorm = Experiment_utils.unNormalize_vel(test_cond, self.data.scaler)
        #self.data.save_animation_witGT(test_cond_unNorm,test_init_unNorm,test_gt_unNorm,os.path.join(self.log_dir,f'GT_vs_init'))
        
        # """end-effector condition or not"""
        # ee_erase_idx =[
        #     0,1,2, 
        #     3,4,5,
        #     6,7,8,
        #     9,10,11,
        #     12,13,14
        # ]
        # ee_idx = [
        #     (5)*3 +0,(5)*3 +1,(5)*3 +2,
        #     (9)*3 +0,(9)*3 +1,(9)*3 +2,
        #     (13)*3 +0,(13)*3 +1,(13)*3 +2,
        #     (17)*3 +0,(17)*3 +1,(17)*3 +2,
        #     (21)*3 +0,(21)*3 +1,(21)*3 +2
        # ]
        """ upper """
        ee_idx_upper = [
            (5)*3 +0,(5)*3 +1,(5)*3 +2,
            (9)*3 +0,(9)*3 +1,(9)*3 +2,
            (13)*3 +0,(13)*3 +1,(13)*3 +2
        ]
        ee_erase_idx_upper =[
            9,10,11,
            12,13,14
        ]

        
        # """ hand """
        # ee_idx_hand = [
        #     (9)*3 +0,(9)*3 +1,(9)*3 +2,
        #     (13)*3 +0,(13)*3 +1,(13)*3 +2
        # ]
        # ee_erase_idx_hand =[
        #     0,1,2, 
        #     9,10,11,
        #     12,13,14
        # ]

        # """ foot """
        # ee_idx_foot = [
        #     (17)*3 +0,(17)*3 +1,(17)*3 +2,
        #     (21)*3 +0,(21)*3 +1,(21)*3 +2
        # ]
        # ee_erase_idx_foot =[
        #     0,1,2, 
        #     3,4,5,
        #     6,7,8
        # ]

        ee_idx = ee_idx_upper
        ee_erase_idx = ee_erase_idx_upper


        if( 15 - len(ee_erase_idx) != len(ee_idx)):
            print("wrong! wrong! wrong! : check the erase and cotain end effector ")
        
        anim_clip = Experiment_utils.unNormalize_motion(test_gt,self.data.scaler)
        # calculate score
        motion_clip = anim_clip[...,:66] /1.7
        # get score
        apd_score_GT = Experiment_utils.calculate_APD(motion_clip)
        print(f"apd_score_ofGT",apd_score_GT)

        # evaluate Samples result
        apd_score_forGT = np.zeros((N))
        for i in range(0,N):
            # random sample
            sampled_z_all = graph.loss_fn_GMM.sample_all(nBatch)
            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nBatch,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device) # (B,10,F)
            # random sample
            #sampled_z_all = graph.distribution.sample((nBatch,66,1), eps_std =1.0)
            #Total_samples[i] = self.generate_motion_label(graph, graph_cond, test_cond, seqlen, sampled_z_all,test_env,test_gt_all=test_gt,ee_erase_idx=ee_erase_idx)
            Total_samples[i] = self.generate_motion_label(graph, graph_cond, test_cond, seqlen, sampled_z_all,test_env)

            Total_samples_forGT[1] = Total_samples[i]
            apd_score_forGT[i] = Experiment_utils.get_motion_APD_Score(Total_samples_forGT[0:2],totalClips=2,scaler=self.data.scaler) # apd between GT
            print(f"{i}_apd_score_withGT",apd_score_forGT[i])

        print(f"apd_score_withGT",np.mean(apd_score_forGT))
            

        
        # save animation 
        # for i in range(0,N):
        #     test_samp_unNorm = Experiment_utils.unNormalize_motion(Total_samples[i], self.data.scaler)
        #     self.data.save_animation_witGT(test_cond_unNorm,test_samp_unNorm,test_gt_unNorm,os.path.join(self.log_dir,f'GT_vs_Sample_{i}'))
        
        # evaluate result
        apd_score = Experiment_utils.get_motion_APD_Score(Total_samples,N,self.data.scaler)
        fd_score = Experiment_utils.get_motion_FD_Score(Total_samples, test_gt, N, self.data.scaler)
        # evaluate end-effector 
        ee_score = Experiment_utils.get_motion_EED_Score(Total_samples, test_gt, N, self.data.scaler,ee_idx)
        return apd_score
    
    def generate_test_eval_noGMM(self,graph,graph_cond,seqlen,test_gt,test_cond,test_env,N=10):

        print("generate_sample")
        graph.eval()
        graph_cond.eval()
    
        # generate N = 10 samples
        # test_gt = test_gt[0:1,...]
        # test_cond = test_cond[0:1,...]
        # test_env = test_env[0:1,...]
        nBatch, nTimesteps, nFeatures = test_gt.shape
        Total_samples = np.zeros((N,nBatch,nTimesteps,nFeatures)) # (N, B, T,F)
        Total_samples_forGT = np.zeros((2,nBatch,nTimesteps,nFeatures)) # (N, B, T,F)
        Total_samples_forGT[0] = test_gt

        # evaluate GT result
        #Total_samples[0] = test_gt
        #apd_score = Experiment_utils.get_motion_APD_Score(Total_samples[0:2],totalClips=2,scaler=self.data.scaler)
        
        # save animation
        #test_gt_unNorm = Experiment_utils.unNormalize_motion(test_gt, self.data.scaler)
        #test_init_unNorm = Experiment_utils.unNormalize_motion(Total_samples[1], self.data.scaler)
        #test_cond_unNorm = Experiment_utils.unNormalize_vel(test_cond, self.data.scaler)
        #self.data.save_animation_witGT(test_cond_unNorm,test_init_unNorm,test_gt_unNorm,os.path.join(self.log_dir,f'GT_vs_init'))
        
        # end-effector condition or not
        # ee_erase_idx =[
        #     0,1,2, 
        #     3,4,5,
        #     6,7,8,
        #     9,10,11,
        #     12,13,14
        # ]
        # ee_idx = [
        #     (5)*3 +0,(5)*3 +1,(5)*3 +2,
        #     (9)*3 +0,(9)*3 +1,(9)*3 +2,
        #     (13)*3 +0,(13)*3 +1,(13)*3 +2,
        #     (17)*3 +0,(17)*3 +1,(17)*3 +2,
        #     (21)*3 +0,(21)*3 +1,(21)*3 +2
        # ]
        """ upper """
        ee_idx_upper = [
            (5)*3 +0,(5)*3 +1,(5)*3 +2,
            (9)*3 +0,(9)*3 +1,(9)*3 +2,
            (13)*3 +0,(13)*3 +1,(13)*3 +2
        ]
        ee_erase_idx_upper =[
            9,10,11,
            12,13,14
        ]
        """ upper body """
        ee_idx_upperbody = list(range(0, 42))
        ee_erase_idx_upperbody =list(range(42, 66))

        # """ hand """
        # ee_idx_hand = [
        #     (9)*3 +0,(9)*3 +1,(9)*3 +2,
        #     (13)*3 +0,(13)*3 +1,(13)*3 +2
        # ]
        # ee_erase_idx_hand =[
        #     0,1,2, 
        #     9,10,11,
        #     12,13,14
        # ]

        # """ foot """
        # ee_idx_foot = [
        #     (17)*3 +0,(17)*3 +1,(17)*3 +2,
        #     (21)*3 +0,(21)*3 +1,(21)*3 +2
        # ]
        # ee_erase_idx_foot =[
        #     0,1,2, 
        #     3,4,5,
        #     6,7,8
        # ]

        ee_idx = ee_idx_upper
        ee_erase_idx = ee_erase_idx_upper


        if( 15 - len(ee_erase_idx) != len(ee_idx)):
            print("wrong! wrong! wrong! : check the erase and cotain end effector ")
        
        anim_clip = Experiment_utils.unNormalize_motion(test_gt,self.data.scaler)
        # calculate score
        motion_clip = anim_clip[...,:66] /1.7
        # get score
        apd_score_GT = Experiment_utils.calculate_APD(motion_clip)
        print(f"apd_score_ofGT",apd_score_GT)

        # evaluate Samples result
        apd_score_forGT = np.zeros((N))
        for i in range(0,N):
            # random sample
            # sampled_z_all = graph.loss_fn_GMM.sample_all(nBatch)
            # sampled_z_all = sampled_z_all.unsqueeze(0)
            # sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nBatch,-1)
            # sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device) # (B,10,F)
            # random sample
            sampled_z_all = graph.distribution.sample((nBatch,1,66), eps_std =1.0)
            #Total_samples[i] = self.generate_motion_woGMM(graph, graph_cond, test_cond, seqlen, sampled_z_all,test_env,test_gt_all=test_gt,ee_erase_idx=ee_erase_idx)
            Total_samples[i] = self.generate_motion_woGMM(graph, graph_cond, test_cond, seqlen, sampled_z_all,test_env)

            Total_samples_forGT[1] = Total_samples[i]
            apd_score_forGT[i] = Experiment_utils.get_motion_APD_Score(Total_samples_forGT[0:2],totalClips=2,scaler=self.data.scaler) # apd between GT
            print(f"{i}_apd_score_withGT",apd_score_forGT[i])

        print(f"apd_score_withGT",np.mean(apd_score_forGT))
            

        
        # save animation 
        # for i in range(0,N):
        #     test_samp_unNorm = Experiment_utils.unNormalize_motion(Total_samples[i], self.data.scaler)
        #     self.data.save_animation_witGT(test_cond_unNorm,test_samp_unNorm,test_gt_unNorm,os.path.join(self.log_dir,f'GT_vs_Sample_{i}'))
        
        # evaluate result
        apd_score = Experiment_utils.get_motion_APD_Score(Total_samples,N,self.data.scaler)
        fd_score = Experiment_utils.get_motion_FD_Score(Total_samples, test_gt, N, self.data.scaler)
        # evaluate end-effector 
        ee_score = Experiment_utils.get_motion_EED_Score(Total_samples, test_gt, N, self.data.scaler,ee_idx)
        return apd_score

    def generate_test_eval(self,graph,graph_cond,seqlen,test_gt,test_cond,test_env,N=10):

        print("generate_sample")
        graph.eval()
        graph_cond.eval()
    
        # generate N = 10 samples
        # test_gt = test_gt[0:1,...]
        # test_cond = test_cond[0:1,...]
        # test_env = test_env[0:1,...]
        nBatch, nTimesteps, nFeatures = test_gt.shape
        Total_samples = np.zeros((N,nBatch,nTimesteps,nFeatures)) # (N, B, T,F)
        Total_samples_forGT = np.zeros((2,nBatch,nTimesteps,nFeatures)) # (N, B, T,F)
        Total_samples_forGT[0] = test_gt

        # evaluate GT result
        #Total_samples[0] = test_gt
        #apd_score = Experiment_utils.get_motion_APD_Score(Total_samples[0:2],totalClips=2,scaler=self.data.scaler)
        
        # save animation
        #test_gt_unNorm = Experiment_utils.unNormalize_motion(test_gt, self.data.scaler)
        #test_init_unNorm = Experiment_utils.unNormalize_motion(Total_samples[1], self.data.scaler)
        #test_cond_unNorm = Experiment_utils.unNormalize_vel(test_cond, self.data.scaler)
        #self.data.save_animation_witGT(test_cond_unNorm,test_init_unNorm,test_gt_unNorm,os.path.join(self.log_dir,f'GT_vs_init'))
        
        # """end-effector condition or not"""
        # ee_erase_idx =[
        #     0,1,2, 
        #     3,4,5,
        #     6,7,8,
        #     9,10,11,
        #     12,13,14
        # ]
        # ee_idx = [
        #     (5)*3 +0,(5)*3 +1,(5)*3 +2,
        #     (9)*3 +0,(9)*3 +1,(9)*3 +2,
        #     (13)*3 +0,(13)*3 +1,(13)*3 +2,
        #     (17)*3 +0,(17)*3 +1,(17)*3 +2,
        #     (21)*3 +0,(21)*3 +1,(21)*3 +2
        # ]
        """ upper """
        ee_idx_upper = [
            (5)*3 +0,(5)*3 +1,(5)*3 +2,
            (9)*3 +0,(9)*3 +1,(9)*3 +2,
            (13)*3 +0,(13)*3 +1,(13)*3 +2
        ]
        ee_erase_idx_upper =[
            9,10,11,
            12,13,14
        ]

        
        # """ hand """
        # ee_idx_hand = [
        #     (9)*3 +0,(9)*3 +1,(9)*3 +2,
        #     (13)*3 +0,(13)*3 +1,(13)*3 +2
        # ]
        # ee_erase_idx_hand =[
        #     0,1,2, 
        #     9,10,11,
        #     12,13,14
        # ]

        # """ foot """
        # ee_idx_foot = [
        #     (17)*3 +0,(17)*3 +1,(17)*3 +2,
        #     (21)*3 +0,(21)*3 +1,(21)*3 +2
        # ]
        # ee_erase_idx_foot =[
        #     0,1,2, 
        #     3,4,5,
        #     6,7,8
        # ]

        ee_idx = ee_idx_upper
        ee_erase_idx = ee_erase_idx_upper


        if( 15 - len(ee_erase_idx) != len(ee_idx)):
            print("wrong! wrong! wrong! : check the erase and cotain end effector ")
        
        anim_clip = Experiment_utils.unNormalize_motion(test_gt,self.data.scaler)
        # calculate score
        motion_clip = anim_clip[...,:66] /1.7
        # get score
        apd_score_GT = Experiment_utils.calculate_APD(motion_clip)
        print(f"apd_score_ofGT",apd_score_GT)

        # evaluate Samples result
        apd_score_forGT = np.zeros((N))
        for i in range(0,N):
            # random sample
            sampled_z_all = graph.loss_fn_GMM.sample_all(nBatch)
            sampled_z_all = sampled_z_all.unsqueeze(0)
            sampled_z_all = sampled_z_all.reshape(graph.means.shape[0],nBatch,-1)
            sampled_z_all = sampled_z_all.permute(1,0,2).to(self.data_device) # (B,10,F)
            # random sample
            #sampled_z_all = graph.distribution.sample((nBatch,66,1), eps_std =1.0)
            #Total_samples[i] = self.generate_motion(graph, graph_cond, test_cond, seqlen, sampled_z_all,test_env,test_gt_all=test_gt,ee_erase_idx=ee_erase_idx)
            Total_samples[i] = self.generate_motion(graph, graph_cond, test_cond, seqlen, sampled_z_all,test_env)

            Total_samples_forGT[1] = Total_samples[i]
            apd_score_forGT[i] = Experiment_utils.get_motion_APD_Score(Total_samples_forGT[0:2],totalClips=2,scaler=self.data.scaler) # apd between GT
            print(f"{i}_apd_score_withGT",apd_score_forGT[i])

        print(f"apd_score_withGT",np.mean(apd_score_forGT))
            

        
        # save animation 
        # for i in range(0,N):
        #     test_samp_unNorm = Experiment_utils.unNormalize_motion(Total_samples[i], self.data.scaler)
        #     self.data.save_animation_witGT(test_cond_unNorm,test_samp_unNorm,test_gt_unNorm,os.path.join(self.log_dir,f'GT_vs_Sample_{i}'))
        
        # evaluate result
        apd_score = Experiment_utils.get_motion_APD_Score(Total_samples,N,self.data.scaler)
        fd_score = Experiment_utils.get_motion_FD_Score(Total_samples, test_gt, N, self.data.scaler)
        # evaluate end-effector 
        ee_score = Experiment_utils.get_motion_EED_Score(Total_samples, test_gt, N, self.data.scaler,ee_idx)
        return apd_score