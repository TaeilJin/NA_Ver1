import numpy as np
import joblib

def concat_sequence_3(seqlen, data):
    """ 
    Concatenates a sequence of features to one.
    """
    nn,n_timesteps,n_feats = data.shape
    L = n_timesteps-(seqlen-1)
    inds = np.zeros((L, seqlen)).astype(int)

    #create indices for the sequences we want
    rng = np.arange(0, n_timesteps)
    for ii in range(0,seqlen):  
        inds[:, ii] = np.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))])  

    #slice each sample into L sequences and store as new samples 
    cc=data[:,inds,:].copy()

    #print ("cc: " + str(cc.shape))

    #reshape all timesteps and features into one dimention per sample
    dd = cc.reshape((nn, L, seqlen*n_feats))
    #print ("dd: " + str(dd.shape))
    return dd 

def create_Sequence_CondOnly_OutputData3(seqlen,data,condinfo, labelinfo=0):
    # sequence data
    joint_data = data[:,:,:-(condinfo+labelinfo)] # joint data
    if labelinfo is not 0 :
        control_data = data[:,:,-(condinfo+labelinfo):-labelinfo] # control data
        label_data = data[:,:,-labelinfo:] # label data
    else:
        control_data = data[:,:,-(condinfo+labelinfo):] # control data
        label_data = data[:,:,:0] # label data 

    # current pose (output)
    n_frames = joint_data.shape[1]
    new_x = concat_sequence_3(1, joint_data[:,seqlen:n_frames,:])
    new_label = concat_sequence_3(1,label_data[:,seqlen:n_frames,:])
    # control autoreg(10) + control(11 or 1)
    autoreg_control = concat_sequence_3(seqlen +1, control_data)
    single_control = concat_sequence_3(1, control_data[:,seqlen:n_frames,:])
    
    #
    autoreg_seq = concat_sequence_3(seqlen,joint_data[:,:n_frames-1,:])
    autoreg_seq_control = np.concatenate((autoreg_seq,autoreg_control),axis=-1)
    autoreg_seq_single_control = np.concatenate((autoreg_seq,single_control),axis=-1)
    
    return new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control, new_label
   
def save_split_test_data(data,data_root,b_save=False, label_info=0):
    # scaling
    scaled_data_X = data.copy()
    
    if(b_save):
        datafilecnt = 0
        for i_te in range(0, data.shape[0]):
            print("datafilecnt"+str(datafilecnt))
            if label_info == 0 :
                np.savez_compressed(f'{data_root}_scaled_x_{str(datafilecnt)}.npz', clips = scaled_data_X[i_te,:,:])
                np.savez_compressed(f'{data_root}_scaled_label_{str(datafilecnt)}.npz', clips = scaled_data_X[i_te,:,:])
            else:
                np.savez_compressed(f'{data_root}_scaled_x_{str(datafilecnt)}.npz', clips = scaled_data_X[i_te,:,:-label_info])
                np.savez_compressed(f'{data_root}_scaled_label_{str(datafilecnt)}.npz', clips = scaled_data_X[i_te,:,-label_info:])

            datafilecnt += 1    
            
def save_split_trainable_data(seqlen, data, condinfo,labelinfo=0, b_save=False, data_root=None):
    # scaling
    scaled_data_X = data.copy()
   
    if(b_save):
        # trainable data
        new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control, new_label = create_Sequence_CondOnly_OutputData3(seqlen,scaled_data_X,condinfo,labelinfo)
        #save
        datafilecnt_train =0
        for i in range(0,new_x.shape[0]):
            print("datafilecnt_train"+str(datafilecnt_train))
            np.savez_compressed(f'{data_root}_scaled_seqX_{str(datafilecnt_train)}.npz', clips = new_x[i,...])
            np.savez_compressed(f'{data_root}_scaled_seqlabel_{str(datafilecnt_train)}.npz', clips = new_label[i,...])
            
            np.savez_compressed(f'{data_root}_scaled_seqControl_{str(datafilecnt_train)}.npz', clips = autoreg_control[i,...])
            np.savez_compressed(f'{data_root}_scaled_singleControl_{str(datafilecnt_train)}.npz', clips = single_control[i,...])
        
            np.savez_compressed(f'{data_root}_scaled_seqControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_control[i,...])
            np.savez_compressed(f'{data_root}_scaled_singleControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_single_control[i,...])
            
            datafilecnt_train += 1


create_dir_save_trainable_data = "/root/home/project/data/locomotion/2022/loco_env_withLabel"
scaler = joblib.load(f'{create_dir_save_trainable_data}/mixamo.pkl')
train_X = np.load(f"{create_dir_save_trainable_data}/train_scaled_all.npz")['clips'].astype(np.float32)
valid_X = np.load(f"{create_dir_save_trainable_data}/valid_scaled_all.npz")['clips'].astype(np.float32)
test_X = np.load(f"{create_dir_save_trainable_data}/test_scaled_all.npz")['clips'].astype(np.float32)

b_split_data = True
save_split_trainable_data(seqlen = 10, data = train_X, condinfo=3,labelinfo=1, b_save = b_split_data, data_root = f"{create_dir_save_trainable_data}/train")
save_split_trainable_data(seqlen = 10, data = valid_X, condinfo=3,labelinfo=1, b_save = b_split_data, data_root = f"{create_dir_save_trainable_data}/valid")
save_split_test_data(test_X,f"{create_dir_save_trainable_data}/test",b_split_data,label_info=1)