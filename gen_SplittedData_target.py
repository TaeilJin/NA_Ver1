import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def partial_fit(data,scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler.partial_fit(flat)
    
    return scaler
        
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

def gen_scaler(train_X, scaler, b_save =False,data_root =None):
    # scaler update
    scaler = partial_fit(train_X, scaler)
    if(b_save):
        # # scaler 저장하기
        joblib.dump(scaler,os.path.join(data_root,f'{data_root}/mixamo.pkl'))
        # scaler 불러오기
        scaler = joblib.load(os.path.join(data_root,f'{data_root}/mixamo.pkl'))
    return scaler

def get_txtfiles(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.txt')] 
def get_npzfiles(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.npz')] 

def gen_scaler_env(train_X,scaler,n_feature=3,b_pos = False, b_pos_rot = False, b_ori_rot = False, b_save =False,data_root =None):
    # scaler update
    # data -> pose velocity 만
    if(b_ori_rot == True):
        pose_X = train_X[:,:,:69].copy()
        ee_X = train_X[:,:,69:69+30]
        vel_X = train_X[:,:,-4:-1]
        train_X = np.concatenate((pose_X,ee_X,vel_X),axis=-1)
    if(b_pos == True):
        pose_X = train_X[:,:,:66].copy()
        ee_X = train_X[:,:,66:66+30]
        vel_X = train_X[:,:,-4:-1]
        vel_target = train_X[:,:,66+30+4:66+30+4+3]
        train_X = np.concatenate((pose_X,ee_X,vel_target,vel_X),axis=-1)
    if(b_pos_rot == True):
        pose_X = train_X[:,:,:132].copy()
        ee_X = train_X[:,:,132:132+30]
        vel_X = train_X[:,:,-4:-1]
        train_X = np.concatenate((pose_X,ee_X,vel_X),axis=-1)
    # scaler partial fit
    scaler = partial_fit(train_X, scaler)
    if(b_save):
        # # scaler 저장하기
        joblib.dump(scaler,os.path.join(data_root,f'{data_root}/mixamo.pkl'))
        # scaler 불러오기
        scaler = joblib.load(os.path.join(data_root,f'{data_root}/mixamo.pkl'))
    return scaler    

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

def create_Sequence_CondOnly_OutputData3(seqlen,data,
num_ee,num_vel,num_env,num_label,scaler,f_contact=0):
        
    # sequence data joint(66) + ee(30)+ fcontact(4)+ target_vel(3) + pre_env(2640)+ cur_env(2640) + vel(3) + label(1)
    joint_data = data[:,:,:-(num_ee + f_contact + num_vel+num_env +num_env+ num_vel + num_label)].copy() # joint data
    joint_num = joint_data.shape[-1]
    if num_env is not 0 :
        ee_data = data[:,:,joint_num:(joint_num+num_ee)].copy() # control data
        vel_data = data[:,:,(joint_num + num_ee + f_contact + num_vel + num_env +num_env):(joint_num + num_ee + f_contact + num_vel + num_env +num_env+num_vel) ].copy()
        f_contact_data = data[:,:,(joint_num+num_ee):(joint_num+num_ee+f_contact)].copy()
        label_data = data[:,:,-num_label:].copy() # label data
    else:
        vel_data = data[:,:,-(num_vel+num_label):].copy() # control data
        label_data = data[:,:,:0].copy() # label data 
    # 
    env_data = data[:,:,(joint_num+num_ee + f_contact +num_vel+num_env):(joint_num+num_ee + f_contact +num_vel+num_env+num_env)].copy()
    prev_env_data = data[:,:,(joint_num+num_ee + f_contact +num_vel):(joint_num+num_ee + f_contact +num_vel+num_env)].copy()
    target_vel_data = data[:,:,(joint_num+num_ee + f_contact):(joint_num+num_ee + f_contact +num_vel)].copy()
    
    # scaling
    scale_data = np.concatenate((joint_data,ee_data,target_vel_data,vel_data),axis=-1)
    scale_data = standardize(scale_data,scaler)
    joint_data = scale_data[...,:joint_num]
    ee_data = scale_data[...,joint_num:-(num_vel*2)]
    target_vel_data = scale_data[...,joint_num+30:-num_vel]
    vel_data = scale_data[...,-(num_vel):]
    
    # current pose (output)
    n_frames = joint_data.shape[1]
    new_x = concat_sequence_3(1, joint_data[:,seqlen:n_frames,:])
    new_env = concat_sequence_3(1, env_data[:,seqlen:n_frames,:])
    new_label = concat_sequence_3(1,label_data[:,seqlen:n_frames,:])
    new_ee = concat_sequence_3(1,ee_data[:,seqlen:n_frames,:])
    new_fcontact = concat_sequence_3(1,f_contact_data[:,seqlen:n_frames,:])
    prev_env = concat_sequence_3(1, prev_env_data[:,seqlen:n_frames,:])
    
    # control autoreg(10) + control(11 or 1)
    autoreg_target_vel = concat_sequence_3(seqlen +1, target_vel_data)
    single_target_vel = concat_sequence_3(1, target_vel_data[:,seqlen:n_frames,:])
    
    autoreg_control = concat_sequence_3(seqlen +1, vel_data)
    single_control = concat_sequence_3(1, vel_data[:,seqlen:n_frames,:])
    

    # test_scale = np.concatenate((new_x,new_ee,new_env,single_control,new_label),axis=-1)[1]
    # np.savetxt("ScaledData.txt",test_scale,delimiter=" ")
    #
    autoreg_seq = concat_sequence_3(seqlen,joint_data[:,:n_frames-1,:])
    autoreg_seq_control = np.concatenate((autoreg_seq,autoreg_control),axis=-1)
    autoreg_seq_single_control = np.concatenate((autoreg_seq,single_control),axis=-1)
    
    # joint ( past 10 ) root vel (past 10) target vel (past 10 ) target vel (current) previous env
    autoreg_desired_vel = np.concatenate((autoreg_seq,autoreg_control[...,:-3],autoreg_target_vel,prev_env),axis=-1)
    
    return new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control, new_ee,new_env,new_label, new_fcontact, autoreg_desired_vel

def save_split_test_data(scaler,data,data_root,filecnt_start, num_ee, num_vel, num_env, foot_contact = 2 ,num_label = 1, b_save=False):
    # scaling
    # sequence data joint(66) + ee(30)+ fcontact(4)+ target_vel(3) + pre_env(2640)+ cur_env(2640) + vel(3) + label(1)
    joint_data = data[:,:,:-(num_ee + foot_contact+ num_vel+ num_env+num_env + num_vel+num_label)] # joint data
    joint_num = joint_data.shape[-1]
    if num_env is not 0 :
        ee_data = data[:,:,joint_num:(joint_num+num_ee)].copy() # control data
        vel_data = data[:,:,(joint_num + num_ee + foot_contact + num_vel + num_env +num_env):(joint_num + num_ee + foot_contact + num_vel + num_env +num_env+num_vel) ].copy()
        f_contact_data = data[:,:,(joint_num+num_ee):(joint_num+num_ee+foot_contact)].copy()
        label_data = data[:,:,-num_label:].copy() # label data
    prev_env_data = data[:,:,(joint_num+num_ee + foot_contact +num_vel):(joint_num+num_ee + foot_contact +num_vel+num_env)].copy()
    target_vel_data = data[:,:,(joint_num+num_ee + foot_contact):(joint_num+num_ee + foot_contact +num_vel)].copy()
    env_data = data[:,:,(joint_num+num_ee + foot_contact +num_vel+num_env):(joint_num+num_ee + foot_contact +num_vel+num_env+num_env)].copy()
    
    
    # scaling
    scale_data = np.concatenate((joint_data,ee_data,target_vel_data,vel_data),axis=-1)
    scale_data = standardize(scale_data,scaler)
    
    joint_data = scale_data[...,:joint_num]
    ee_data = scale_data[...,joint_num:-(num_vel*2)]
    target_vel_data = scale_data[...,joint_num+30:-num_vel]
    vel_data = scale_data[...,-(num_vel):]


    if(b_save):
        datafilecnt = filecnt_start
        for i_te in range(0, data.shape[0]):
            print("datafilecnt"+str(datafilecnt))
            
            np.savez_compressed(f'{data_root}_scaled_x_{str(datafilecnt)}.npz', clips = joint_data[i_te,:,:])
            np.savez_compressed(f'{data_root}_scaled_vel_{str(datafilecnt)}.npz', clips = vel_data[i_te,:,:])
            np.savez_compressed(f'{data_root}_scaled_ee_{str(datafilecnt)}.npz', clips = ee_data[i_te,:,:])
            np.savez_compressed(f'{data_root}_scaled_label_{str(datafilecnt)}.npz', clips = label_data[i_te,:,])
            np.savez_compressed(f'{data_root}_scaled_env_{str(datafilecnt)}.npz', clips = env_data[i_te,:,:])
            np.savez_compressed(f'{data_root}_scaled_fcontact_{str(datafilecnt)}.npz', clips = f_contact_data[i_te,:,:])
            np.savez_compressed(f'{data_root}_scaled_preenv_{str(datafilecnt)}.npz', clips = prev_env_data[i_te,:,:])
            np.savez_compressed(f'{data_root}_scaled_tarVel_{str(datafilecnt)}.npz', clips = target_vel_data[i_te,:,:])
            
            datafilecnt += 1  
    return datafilecnt  


def save_split_trainable_data(scaler, seqlen, data, filecnt_train, num_ee, num_vel, num_envsample, foot_contact =0, labelinfo=0, b_save=False, data_root=None):
    # scaling
    scaled_data_X = data.copy()
    
    if(b_save):
        # trainable data 
        new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control, new_ee,new_env,new_label, new_foot, root_estimate_input = create_Sequence_CondOnly_OutputData3(seqlen,scaled_data_X,num_ee,num_vel,num_envsample,labelinfo,
        scaler,f_contact=foot_contact)
        #save
        datafilecnt_train = filecnt_train
        for i in range(0,new_x.shape[0]):
            print("datafilecnt_train"+str(datafilecnt_train))
            np.savez_compressed(f'{data_root}_scaled_x_{str(datafilecnt_train)}.npz', clips = new_x[i,...])
            np.savez_compressed(f'{data_root}_scaled_env_{str(datafilecnt_train)}.npz', clips = new_env[i,...])
            np.savez_compressed(f'{data_root}_scaled_label_{str(datafilecnt_train)}.npz', clips = new_label[i,...])
            np.savez_compressed(f'{data_root}_scaled_ee_{str(datafilecnt_train)}.npz', clips = new_ee[i,...])
            np.savez_compressed(f'{data_root}_scaled_fcontact_{str(datafilecnt_train)}.npz', clips = new_foot[i,...])
            
            np.savez_compressed(f'{data_root}_scaled_seqControl_{str(datafilecnt_train)}.npz', clips = autoreg_control[i,...])
            np.savez_compressed(f'{data_root}_scaled_singleControl_{str(datafilecnt_train)}.npz', clips = single_control[i,...])
        
            np.savez_compressed(f'{data_root}_scaled_seqControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_control[i,...])
            np.savez_compressed(f'{data_root}_scaled_singleControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_single_control[i,...])
            
            np.savez_compressed(f'{data_root}_scaled_root_estimate_input_{str(datafilecnt_train)}.npz', clips = root_estimate_input[i,...])
        

            datafilecnt_train += 1
    
    return datafilecnt_train

def gen_learnable_data_inFolder(data_root,save_root,env_dim=2640,b_split_data=False):
        
    bvh_files = get_npzfiles(data_root) # 198 2830*3 2830 3 1

    data_X = np.load(bvh_files[0])['clips'].astype(np.float32)
    #data_X = create_trainable_data(data_X,3,env_dim,True)
    # if bvh_files[0].startswith(f"{data_root}/trans_") or bvh_files[0].startswith(f"{data_root}/mani_"):
    #     data_X = data_X.repeat(3,axis=0)
    for i in range(1,len(bvh_files)):
        print ('processing %i of %i (%s)' % (i, len(bvh_files),bvh_files[i]))
        bvh_file = bvh_files[i]
        # load clip
        clip_data_X = np.load(bvh_file)['clips'].astype(np.float32)
        #clip_data_X = create_trainable_data(clip_data_X,3,env_dim,True)
        # if bvh_files[i].startswith(f"{data_root}/trans_") or bvh_files[0].startswith(f"{data_root}/mani_"):
        #     clip_data_X = clip_data_X.repeat(3,axis=0)
        #condatenation
        data_X = np.concatenate((data_X, clip_data_X),axis = 0)
    
    if (b_split_data == False):
        # all data is a train data
        train_X = data_X
        np.savez_compressed(f'{save_root}/train_all.npz', clips = train_X)
        return train_X, np.zeros(1), np.zeros(1)
    else:
        # data_X -> train, valid , test
        train_X, valid_X = train_test_split(data_X, test_size=0.5,random_state=1004) # train valid 나눠서 sequence data로 만듬 
        valid_X, test_X = train_test_split(valid_X, test_size=0.5,random_state=1004) # train valid 나눠서 sequence data로 만듬 
        
        train_X = np.concatenate((train_X,valid_X[30:,...]),axis=0)
        train_X = np.concatenate((train_X,test_X[10:,...]), axis=0)
        valid_X = valid_X[:30,...]
        test_X = test_X[:10,...]
        
        np.savez_compressed(f'{save_root}/train_all.npz', clips = train_X)
        np.savez_compressed(f'{save_root}/valid_all.npz', clips = valid_X)
        np.savez_compressed(f'{save_root}/test_all.npz', clips = test_X)
        
    return train_X, valid_X, test_X

def create_trainable_data(data, n_feature=3,n_env=2640, b_label=False):
           
    data_copy = data.copy()
    if (b_label == True):
        label_id = 1
        pose_X = data_copy[:,:,:22*n_feature].copy() 
        vel_X = data_copy[:,:,-4:-1]
        env_occupancy_X = data_copy[:,:,-(n_env+4):-(4)]
        label_Y = data[...,-(label_id):].copy()

    data_X = np.concatenate((pose_X,vel_X),axis=-1)
    
    if(b_label == True):
        data_X = np.concatenate((data_X,env_occupancy_X),axis=-1)
        data_X = np.concatenate((data_X,label_Y),axis=-1)
    
    return data_X

def gen_Train_Valid_Test_Data(data_root,scaler_root,b_split_data=False,b_label=False, b_first=False,b_pos = False, b_pos_rot = False, b_ori_rot = False):
    
    #training_data
    create_dir_save_trainable_data = f'{data_root}/npz'
    if not os.path.exists(create_dir_save_trainable_data):
        os.makedirs(create_dir_save_trainable_data)

    #scaling 할 데이터 만들기
    env_dim = 2640
    train_X, valid_X, test_X = gen_learnable_data_inFolder(f'{data_root}',create_dir_save_trainable_data,env_dim,b_split_data)
    
    print("data_root: ", data_root)
    print("init scaler: ", b_first)

    # scaler 만들기
    if b_first == True :
        scaler = StandardScaler()
        scaler = gen_scaler_env(train_X,scaler,3,b_pos=b_pos, b_pos_rot=b_pos_rot, b_ori_rot= b_ori_rot, b_save = True,data_root = f'{scaler_root}/npz')
        
    if b_first == False :
        scaler = joblib.load(os.path.join(data_root,f'{scaler_root}/npz/mixamo.pkl'))
        scaler = gen_scaler_env(train_X,scaler,3,b_pos=b_pos, b_pos_rot=b_pos_rot, b_ori_rot= b_ori_rot, b_save = True,data_root = f'{scaler_root}/npz')
        

#--get train scaler
b_pos = True
b_pos_rot = False
b_ori_rot = False

ind_loco = 0
ind_mani = 5
ind_trans = 11

scaler_save_root = "/root/home/project/Data/root_position_target_all"

# #loco
# data_save_root = "/root/home/project/Data/sliced_root_position_target/loco"
# for i in range(0,10):
#     if i == ind_loco:
#         gen_Train_Valid_Test_Data(f"{data_save_root}/sep{i}", scaler_save_root,b_split_data=True, b_label=True,b_first=True,b_pos=b_pos,b_pos_rot=b_pos_rot,b_ori_rot=b_ori_rot)
#     else:
#         gen_Train_Valid_Test_Data(f"{data_save_root}/sep{i}", scaler_save_root,b_split_data=False, b_label=True,b_first=False,b_pos=b_pos,b_pos_rot=b_pos_rot,b_ori_rot=b_ori_rot)
# #mani
# data_save_root = "/root/home/project/Data/sliced_root_position_target/mani"
# for i in range(0,10):
#     if i == ind_mani:
#         gen_Train_Valid_Test_Data(f"{data_save_root}/sep{i}", scaler_save_root,b_split_data=True, b_label=True,b_first=False,b_pos=b_pos,b_pos_rot=b_pos_rot,b_ori_rot=b_ori_rot)
#     else:
#         gen_Train_Valid_Test_Data(f"{data_save_root}/sep{i}", scaler_save_root,b_split_data=False, b_label=True,b_first=False,b_pos=b_pos,b_pos_rot=b_pos_rot,b_ori_rot=b_ori_rot)
# #trans
# data_save_root = "/root/home/project/Data/sliced_root_position_target/trans"
# for i in range(0,20):
#     if i == ind_trans:
#         gen_Train_Valid_Test_Data(f"{data_save_root}/sep{i}", scaler_save_root,b_split_data=True, b_label=True,b_first=False,b_pos=b_pos,b_pos_rot=b_pos_rot,b_ori_rot=b_ori_rot)
#     else:
#         gen_Train_Valid_Test_Data(f"{data_save_root}/sep{i}", scaler_save_root,b_split_data=False, b_label=True,b_first=False,b_pos=b_pos,b_pos_rot=b_pos_rot,b_ori_rot=b_ori_rot)


print("after scaler updated")
# #--split data
data_output_root = f"{scaler_save_root}/npz"
if not os.path.exists(data_output_root):
    os.makedirs(data_output_root)
scaler = joblib.load(f'{data_output_root}/mixamo.pkl')

mean = scaler.mean_
vars = scaler.scale_

if (b_pos == True):
    mean_pos = mean[:66]
    vars_pos = vars[:66]
    
    mean_ee = mean[66: 66 + 30]
    vars_ee = vars[66: 66 + 30]

    mean_target = mean[66+30:66+30+3]
    vars_target = vars[66+30:66+30+3]
    
    mean_vel = mean[-3:]
    vars_vel = vars[-3:]

    

np.savetxt(f'{data_output_root}/mean_pos.txt',mean_pos,delimiter=" ")
np.savetxt(f'{data_output_root}/vars_pos.txt',vars_pos,delimiter=" ")

np.savetxt(f'{data_output_root}/mean_ee.txt',mean_ee,delimiter=" ")
np.savetxt(f'{data_output_root}/vars_ee.txt',vars_ee,delimiter=" ")

np.savetxt(f'{data_output_root}/mean_vel.txt',mean_vel,delimiter=" ")
np.savetxt(f'{data_output_root}/vars_vel.txt',vars_vel,delimiter=" ")

np.savetxt(f'{data_output_root}/mean_target_vel.txt',mean_target,delimiter=" ")
np.savetxt(f'{data_output_root}/vars_target_vel.txt',vars_target,delimiter=" ")


print("generate valid test data")
env_dim = 2640
start_valid_num = 0
start_test_num = 0
#loco
data_save_root = "/root/home/project/Data/sliced_root_position_target/loco"
test_data_root = f"{data_save_root}/sep{ind_loco}"
valid_X = np.load(f"{test_data_root}/npz/valid_all.npz")['clips'].astype(np.float32)
test_X = np.load(f"{test_data_root}/npz/test_all.npz")['clips'].astype(np.float32)

start_valid_num = save_split_trainable_data(scaler,seqlen = 10, data = valid_X, filecnt_train=start_valid_num,num_ee=30, num_vel=3,num_envsample=env_dim,foot_contact = 4 ,labelinfo=1,b_save=True, data_root = f"{data_output_root}/valid")
start_test_num = save_split_test_data(scaler,test_X,f"{data_output_root}/test",filecnt_start = start_test_num,num_ee=30, num_vel=3,num_env=env_dim,foot_contact = 4 ,num_label=1,b_save=True)

#mani
data_save_root = "/root/home/project/Data/sliced_root_position_target/mani"
test_data_root = f"{data_save_root}/sep{ind_mani}"
valid_X = np.load(f"{test_data_root}/npz/valid_all.npz")['clips'].astype(np.float32)
test_X = np.load(f"{test_data_root}/npz/test_all.npz")['clips'].astype(np.float32)
env_dim = 2640
start_valid_num = save_split_trainable_data(scaler,seqlen = 10, data = valid_X, filecnt_train=start_valid_num,num_ee=30, num_vel=3,num_envsample=env_dim,foot_contact = 4 ,labelinfo=1,b_save=True, data_root = f"{data_output_root}/valid")
start_test_num = save_split_test_data(scaler,test_X,f"{data_output_root}/test",filecnt_start = start_test_num,num_ee=30, num_vel=3,num_env=env_dim,foot_contact = 4 ,num_label=1,b_save=True)

#trans
data_save_root = "/root/home/project/Data/sliced_root_position_target/trans"
test_data_root = f"{data_save_root}/sep{ind_trans}"
valid_X = np.load(f"{test_data_root}/npz/valid_all.npz")['clips'].astype(np.float32)
test_X = np.load(f"{test_data_root}/npz/test_all.npz")['clips'].astype(np.float32)
env_dim = 2640
start_valid_num = save_split_trainable_data(scaler,seqlen = 10, data = valid_X, filecnt_train=start_valid_num,num_ee=30, num_vel=3,num_envsample=env_dim,foot_contact = 4 ,labelinfo=1,b_save=True, data_root = f"{data_output_root}/valid")
start_test_num = save_split_test_data(scaler,test_X,f"{data_output_root}/test",filecnt_start = start_test_num,num_ee=30, num_vel=3,num_env=env_dim,foot_contact = 4 ,num_label=1,b_save=True)

print("generate train data")

start_num =0
#loco
data_save_root = "/root/home/project/Data/sliced_root_position_target/loco"
for i in range(0,10):
    data_root = f"{data_save_root}/sep{i}"
    train_X = np.load(f"{data_root}/npz/train_all.npz")['clips'].astype(np.float32)
    start_num = save_split_trainable_data(scaler,seqlen = 10, data = train_X, filecnt_train=start_num,num_ee=30, num_vel=3,num_envsample=env_dim,foot_contact = 4 ,labelinfo=1,b_save=True, data_root = f"{data_output_root}/train")
#mani
data_save_root = "/root/home/project/Data/sliced_root_position_target/mani"
for i in range(0,10):
    data_root = f"{data_save_root}/sep{i}"
    train_X = np.load(f"{data_root}/npz/train_all.npz")['clips'].astype(np.float32)
    start_num = save_split_trainable_data(scaler,seqlen = 10, data = train_X, filecnt_train=start_num,num_ee=30, num_vel=3,num_envsample=env_dim,foot_contact = 4 ,labelinfo=1,b_save=True, data_root = f"{data_output_root}/train")
#loco
data_save_root = "/root/home/project/Data/sliced_root_position_target/trans"
for i in range(0,20):
    data_root = f"{data_save_root}/sep{i}"
    train_X = np.load(f"{data_root}/npz/train_all.npz")['clips'].astype(np.float32)
    start_num = save_split_trainable_data(scaler,seqlen = 10, data = train_X, filecnt_train=start_num,num_ee=30, num_vel=3,num_envsample=env_dim,foot_contact = 4 ,labelinfo=1,b_save=True, data_root = f"{data_output_root}/train")

