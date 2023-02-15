from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R
from frechetdist import frdist
import numpy as np

def inv_standardize(data, scaler):      
    shape = data.shape
    if len(shape) == 2 : 
        scaled = scaler.inverse_transform(data)
    else:
        flat = data.reshape((shape[0]*shape[1], shape[2]))
        scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

def Normalize_motion(pose_data,scaler):
    shape = pose_data.shape
    std = scaler.scale_[:66]
    std[std<1e-3] = 1e-3
    
    if len(shape) == 2 : 
        scaled = (pose_data - scaler.mean_[:66]) / std
    else:
        flat = pose_data.reshape((shape[0]*shape[1], shape[2]))
        scaled = (flat - scaler.mean_[:66]) / std
        scaled = scaled.reshape(shape)
    return scaled     

def Normalize_vel(vel_data,scaler):
    shape = vel_data.shape
    std = scaler.scale_[-3:]
    std[std<1e-3] = 1e-3
    if len(shape) == 2 : 
        scaled = (vel_data - scaler.mean_[-3:]) / std
    else:
        flat = vel_data.reshape((shape[0]*shape[1], shape[2]))
        scaled = (flat - scaler.mean_[-3:]) / std
        scaled = scaled.reshape(shape)
    return scaled        

def unNormalize_motion(pose_data,scaler):
    shape = pose_data.shape
    if len(shape) == 2 : 
        scaled = scaler.mean_[:66] + pose_data * scaler.scale_[:66]
    else:
        flat = pose_data.reshape((shape[0]*shape[1], shape[2]))
        scaled = scaler.mean_[:66] + flat * scaler.scale_[:66]
        scaled = scaled.reshape(shape)
    return scaled     

def unNormalize_vel(vel_data,scaler):
    shape = vel_data.shape
    if len(shape) == 2 : 
        scaled = scaler.mean_[-3:] + vel_data * scaler.scale_[-3:]
    else:
        flat = vel_data.reshape((shape[0]*shape[1], shape[2]))
        scaled = scaler.mean_[-3:] + flat * scaler.scale_[-3:]
        scaled = scaled.reshape(shape)
    return scaled        

def unNormalize_motion(pose_data,mean,scale):
    shape = pose_data.shape
    if len(shape) == 2 : 
        scaled = mean[:66] + pose_data * scale[:66]
    else:
        flat = pose_data.reshape((shape[0]*shape[1], shape[2]))
        scaled = mean[:66] + flat * scale[:66]
        scaled = scaled.reshape(shape)
    return scaled     

def unNormalize_vel(vel_data,mean,scale):
    shape = vel_data.shape
    if len(shape) == 2 : 
        scaled = mean[-3:] + vel_data * scale[-3:]
    else:
        flat = vel_data.reshape((shape[0]*shape[1], shape[2]))
        scaled = mean[-3:] + flat * scale[-3:]
        scaled = scaled.reshape(shape)
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
def calculate_APD(motion_clip):
    
    motion_clip = np.reshape(motion_clip,(motion_clip.shape[0],-1))
    
    dist = pdist(motion_clip)

    apd = dist.mean().item()

    return (apd)
def calculate_FD(gt_clip, motion_clip):
    
    dist = np.zeros((motion_clip.shape[0])) # K
    for nK in range(motion_clip.shape[0]):
        dist[nK] = frdist(gt_clip[0],motion_clip[nK]) # Sampled (T,F)<-> Batch 데이터 (T,F)

    return dist

def caclulate_EED(gt_clip, motion_clip, ee_idx):
        
    motion_clip_ee = motion_clip[:,:,ee_idx]
    gt_clip_ee = gt_clip[:,:,ee_idx]

    # k number's l2 (diff) 
    #diff = motion_clip_ee - gt_clip_ee
    #dist = np.linalg.norm(diff, axis=1).mean(axis=-1)
    #dist.mean()
    dist = np.zeros((motion_clip.shape[0])) # K
    for nK in range(motion_clip.shape[0]):
        dist[nK] = mean_squared_error(motion_clip_ee[nK],gt_clip_ee[0],squared=False) # Sampled (T,F)<-> Batch 데이터 (T,F)

    return dist

def calculate_footVel(gt_clip,motion_clip, ee_idx):
    motion_clip_ee = motion_clip[:,:,ee_idx]
    gt_clip_ee = gt_clip[:,:,ee_idx]
    # k number's l2 (diff) 
    #diff = motion_clip_ee - gt_clip_ee
    #dist = np.linalg.norm(diff, axis=1).mean(axis=-1)
    #dist.mean()
    dist = np.zeros((motion_clip.shape[0])) # K
    for nK in range(motion_clip.shape[0]):
        dist[nK] = mean_squared_error(motion_clip_ee[nK],gt_clip_ee[0],squared=False) # Sampled (T,F)<-> Batch 데이터 (T,F)

    return dist


def gen_world_pos_data(i_joints,i_rootvel):
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


def get_APD_score_batch(motion_data,control_data,height,scaler):
    # after scaler
    animation_data = np.concatenate((motion_data,control_data), axis=-1)
    anim_clip = inv_standardize(animation_data, scaler)
    # calculate score
    motion_clip = anim_clip[...,:-3] /height
    # calc apd
    apd = calculate_APD(motion_clip)
    return apd






def get_APD_Score(control_data, K_motion_data, totalClips, scaler):
    #np.savez("../data/results/test_dropconnect/APD_dropconnect.npz", clips=K_motion_data)
        
    K, nBatch, ntimesteps, feature = K_motion_data.shape
    total_APD_score = np.zeros(nBatch)
    # total_EED_score = np.zeros(nBatch)
    if totalClips != K:
        print("wrong! different motions")
    else :
        for nB in range(nBatch):
            k_motion_data = K_motion_data[:,nB,...] #(K, Timestep, nFeat)
            
            # get single batch 
            batch_control_data = control_data[nB:nB+1,...] #(1, Timestep, nFeat)
            # duplicate k 
            k_control_data = np.repeat(batch_control_data,K,axis=0) #(K, Timestep, nFeat)

            # after scaler
            animation_data = np.concatenate((k_motion_data,k_control_data), axis=2)
            anim_clip = inv_standardize(animation_data, scaler)

            
            # calculate score
            motion_clip = anim_clip[...,:66] /1.7
            # get score
            apd_score = calculate_APD(motion_clip)
            total_APD_score[nB] = np.mean(apd_score)
            
            
        print(f'APD of_{nB}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
        #np.savez(filename + "_APD_score.npz", clips=total_APD_score)
        
    return np.mean(total_APD_score)

def get_motion_APD_Score(K_motion_data, totalClips, scaler):
    #np.savez("../data/results/test_dropconnect/APD_dropconnect.npz", clips=K_motion_data)
        
    K, nBatch, ntimesteps, feature = K_motion_data.shape
    total_APD_score = np.zeros(nBatch)
    # total_EED_score = np.zeros(nBatch)
    if totalClips != K:
        print("wrong! different motions")
    else :
        for nB in range(nBatch):
            k_motion_data = K_motion_data[:,nB,...] #(K, Timestep, nFeat)
             
            anim_clip = unNormalize_motion(k_motion_data,scaler)
            
            # calculate score
            motion_clip = anim_clip[...,:66] /1.7
            # get score
            apd_score = calculate_APD(motion_clip)
            total_APD_score[nB] = np.mean(apd_score)
            
            
        print(f'APD of_{nB}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
        #np.savez(filename + "_APD_score.npz", clips=total_APD_score)
        
    return np.mean(total_APD_score)

def get_motion_FD_Score(K_motion_data, gtClips, totalClips, scaler):
    #np.savez("../data/results/test_dropconnect/APD_dropconnect.npz", clips=K_motion_data)
        
    K, nBatch, ntimesteps, feature = K_motion_data.shape
    total_FD_score = np.zeros(nBatch)
    # total_EED_score = np.zeros(nBatch)
    
    if totalClips != K:
        print("wrong! different motions")
    else :
        for nB in range(nBatch):
            k_motion_data = K_motion_data[:,nB,...] #(K, Timestep, nFeat)
             
            anim_clip = unNormalize_motion(k_motion_data,scaler)
            gt_clip = unNormalize_motion(gtClips[nB:nB+1],scaler)
            # calculate score
            motion_clip = anim_clip[...,:66] /1.7
            # get score
            fd_score = calculate_FD(gt_clip, motion_clip) # (K,T,F) <-> (1,T,F)
            total_FD_score[nB] = np.mean(fd_score)
            
            
        print(f'FD of_{nB}_motion:_{total_FD_score.shape}_:{total_FD_score}_mean:{np.mean(total_FD_score)}')    
        #np.savez(filename + "_APD_score.npz", clips=total_APD_score)
        
    return np.mean(total_FD_score)

def get_motion_EED_Score(K_motion_data, gtClips, totalClips, scaler,ee_idx):
    #np.savez("../data/results/test_dropconnect/APD_dropconnect.npz", clips=K_motion_data)
        
    K, nBatch, ntimesteps, feature = K_motion_data.shape
    total_EED_score = np.zeros(nBatch)
    if totalClips != K:
        print("wrong! different motions")
    else :

        for nB in range(nBatch):
            k_motion_data = K_motion_data[:,nB,...] #(K, Timestep, nFeat)
            
            anim_clip = unNormalize_motion(k_motion_data,scaler) # (K, Timestep, nFeat)
            gt_clip = unNormalize_motion(gtClips[nB:nB+1],scaler) # (1,Timestep,nFeat)
            
            if gt_clip is not None:
                #k_gt_data = np.repeat(gt_clip,K,axis=0) #(K, Timestep, nFeat)                
                # scaling
                gt_clip = gt_clip /1.7
                anim_clip = anim_clip /1.7
                # get score
                eed_score = caclulate_EED(gt_clip,anim_clip,ee_idx)
                total_EED_score[nB] = np.mean(eed_score)
            
        print(f'EED of_{nBatch}_motion:_{total_EED_score.shape}_:{total_EED_score}_mean:{np.mean(total_EED_score)}')    
        #np.savez(filename + "_EED_score.npz", clips=total_EED_score)
    return np.mean(total_EED_score)

def get_motion_Foot_Score(K_motion_data, gtClips, totalClips, scaler,ee_idx):
        #np.savez("../data/results/test_dropconnect/APD_dropconnect.npz", clips=K_motion_data)
            
    K, nBatch, ntimesteps, feature = K_motion_data.shape
    total_Foot_score = np.zeros(nBatch)
    if totalClips != K:
        print("wrong! different motions")
    else :

        for nB in range(nBatch):
            k_motion_data = K_motion_data[:,nB,...] #(K, Timestep, nFeat)
            
            anim_clip = unNormalize_motion(k_motion_data,scaler) # (K, Timestep, nFeat)
            gt_clip = unNormalize_motion(gtClips[nB:nB+1],scaler) # (1,Timestep,nFeat)
            
            if gt_clip is not None:
                #k_gt_data = np.repeat(gt_clip,K,axis=0) #(K, Timestep, nFeat)                
                # scaling
                gt_clip = gt_clip /1.7
                anim_clip = anim_clip /1.7
                # get score
                eed_score = caclulate_EED(gt_clip,anim_clip,ee_idx)
                total_EED_score[nB] = np.mean(eed_score)
        
    print(f'EED of_{nBatch}_motion:_{total_EED_score.shape}_:{total_EED_score}_mean:{np.mean(total_EED_score)}')    
    #np.savez(filename + "_EED_score.npz", clips=total_EED_score)
    return np.mean(total_EED_score)
