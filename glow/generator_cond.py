import re
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from .models_Cond_GMVAE import CondGMVAE
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE

class Generator_Cond(object):
    def __init__(self, data, data_device, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # model relative
        self.data_device = data_device
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        self.data = data
        self.log_dir = log_dir
        self.gaussian_size = 66
        self.num_classes = hparams.Gumbel.num_classes
        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=hparams.Test.batch_size,
                                      num_workers=1,
                                      shuffle=False,
                                      drop_last=True)
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)

        self.parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1

        # train batch (for selecting class number )
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=1,
                                      shuffle=True,
                                      drop_last=True)
    
    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)

    
    def generate_sample_recon(self, graph, gumbel_temp=1.0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["cond"].cpu().numpy()
        label_all = batch["label"]

        condition = 693
        nn,n_timesteps,n_feats = autoreg_all.shape
        features_cond = np.zeros((nn,n_timesteps, condition))
        original_cond = np.zeros((nn,n_timesteps, condition))
        seqlen = 10
        for i in range(0,control_all.shape[1]-seqlen):
            control = control_all[:,i:(i+seqlen+1),:]
            autoreg = autoreg_all[:,i:(i+seqlen),:]
            cond = self.prepare_cond(autoreg.copy(), control.copy())

            nBatch, nFeatures, nTimesteps = cond.shape
            cond = cond.permute(0,2,1)
            cond = cond.reshape(-1,nFeatures).clone().detach()

            out_net = graph.network(cond, gumbel_temp, graph.hard_gumbel)
            reconstructed = out_net['x_rec'].cpu().detach().numpy()
            
            features_cond[:,i+seqlen,:] = reconstructed
            original_cond[:,i+seqlen,:] = cond.cpu().detach().numpy()
        
        loss_rec = ((original_cond[:,seqlen:,:].reshape(-1,condition) - features_cond[:,seqlen:,:].reshape(-1,condition))**2).sum(axis=-1).mean()/693
        print("reconstruction loss (mse) : ",loss_rec)


    def generate_sample_latent_space(self, graph, return_labels, gumbel_temp=1.0):
        print("generate_sample_latent")
        graph.eval()
        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["cond"].cpu().numpy()
        env_all = batch["env"]
        label_all = batch["label"]

        N = 30
        T = autoreg_all.shape[1]
        seqlen = 10
        features = np.zeros((N,T, self.gaussian_size))
        cluster_labels = np.zeros((N,T, 1))
        start_ind = 0

        total_means_byCluster = torch.zeros((self.num_classes,self.gaussian_size))

        for i in range(0,control_all.shape[1]-seqlen):
            # condition
            control = control_all[:,i:(i+seqlen+1),:]
            autoreg = autoreg_all[:,i:(i+seqlen),:]
            env = env_all[:,i+seqlen:(i+seqlen+1),:]

            descriptor = self.prepare_cond(autoreg.copy(), control.copy())
            env = env.permute(0,2,1).to(descriptor.device) # nBatch nFeature, nTimestep
            descriptor = torch.cat((descriptor,env),dim=1)
            # graph outputs
            nBatch, nFeatures, nTimesteps = descriptor.shape
            descriptor = descriptor.permute(0,2,1)
            descriptor = descriptor.reshape(-1,nFeatures).clone().detach()
            
            out = graph.network.inference(descriptor, gumbel_temp, graph.hard_gumbel)
            latent_feat = out['mean'] # B*T , nFeature
            predict_label = out['prob_cat'] # B*T, guassians dim
            _, predicted_labels = torch.max(predict_label, dim=1) # (B*T)
            
            cluster_labels[:,(i+seqlen),0] = predicted_labels.cpu().detach().numpy()

            for c in range(self.num_classes):
                predict_0 = (predicted_labels == c).nonzero(as_tuple=False)[:,0]
                if predict_0.shape[0] > 0:
                    total_means_byCluster[c] = total_means_byCluster[c] + torch.mean(latent_feat[predict_0,:].cpu().detach(),dim=0) # Batch 마다 
                else:
                    total_means_byCluster[c] = total_means_byCluster[c] + 0

            end_ind = min(start_ind + descriptor.size(0), N+1)

            # return true labels
            features[start_ind:end_ind,(i+seqlen),:] = latent_feat[start_ind:end_ind].cpu().detach().numpy()  

        total_means_byCluster = total_means_byCluster / N
        
        # plot only the first 2 dimensions
        features_draw = features[:,seqlen:,:].reshape(-1,self.gaussian_size)
        labels_draw = label_all[:,seqlen:,:].reshape(-1,1).cpu().detach().numpy()
        cluster_draw = cluster_labels[:,seqlen:,:]
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features_draw[:, 0], features_draw[:, 1], c=cluster_draw, marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 3)
        # cluster_colors = np.random.rand(self.num_classes,)

        # model = TSNE(learning_rate=300)
        # transformed = model.fit_transform(total_means_byCluster)

        # plt.scatter(total_means_byCluster[:, 0], total_means_byCluster[:, 1], c=cluster_colors, marker='o',
        #         edgecolor='none', cmap=plt.cm.get_cmap('jet', self.num_classes), s = 30)
        plt.colorbar()
        fig.savefig('latent_space.png')
              
    def generate_sample_accuracy(self, graph, gumbel_temp=1.0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["cond"].cpu().numpy()
    
        label_all = batch["label"]

        nn,n_timesteps,n_feats = autoreg_all.shape
        accuracy_all = np.zeros((nn, n_timesteps, 1))

        seqlen = 10
        true_labels_list = []
        predicted_labels_list = []
        for i in range(0,control_all.shape[1]-seqlen):
            control = control_all[:,i:(i+seqlen+1),:]
            autoreg = autoreg_all[:,i:(i+seqlen),:]
            descriptor = self.prepare_cond(autoreg.copy(), control.copy())

            nBatch, n_timesteps, nFeatures = label_all.shape
            label_idx = label_all[:,(i+seqlen):(i+seqlen+1),:].clone().detach()
            label_true = label_all[:,(i+seqlen):(i+seqlen+1),:].reshape(-1,nFeatures).clone().detach()
            
            _, _, _,_, predict_label = graph(cond=descriptor,label_prob=None, gumbel_temp=gumbel_temp, hard_gumbel= graph.hard_gumbel)
           

            true_labels_list.append(label_true[:,0])
            predicted_labels_list.append(predict_label)  

        true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

        accuracy, nmi = graph.accuracy_test(predicted_labels,true_labels)

        print(f"accuracy_{accuracy}_nmi_{nmi}")

        return accuracy, nmi

    def save_pose_inCluster(self, graph, gumbel_temp=1.0, step=0):
        progress = tqdm(self.data_loader)
        num_total_data = self.train_dataset.__len__()

        total_num_inCluster = np.zeros(graph.num_classes)
        
        # train data 불러오기
        for i_batch, batch in enumerate(progress):

            for k in batch:
                batch[k] = batch[k].to(self.data_device)
                
            # batch 에 대해 graph 넣어서 각각의 label 을 얻는다.
            x = batch["x"]          
            cond = batch["cond"]
            ee_cond = batch["ee_gt"]
            label = batch["label"]
            descriptor = batch["descriptor"]
            # condition encoder -> prior decoding
            with torch.no_grad():
                graph.eval()
                nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach().to(self.data_device)

                out_net  = graph.network(des_enc,gumbel_temp,0.0)
                prob = out_net['prob_cat'] # (B*Timestep, 10)

                
                # for accuracy test "load 확인용"
                _, predicted_labels = torch.max(out_net['logits'], dim=1) # (B*T)

                batch_num_inCluster = np.zeros(graph.num_classes)
                for i in range(num_cls):
                    predict_0 = (predicted_labels == i).nonzero(as_tuple=False)[:,0]
                    #
                    x_batch = x.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach() #(B*T,nFeats)
                    c_batch = cond.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach() #(B*T, nFeats)
                    label_batch = label.reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach()
                    ee_batch    = ee_cond.reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach()

                    x_0 = x_batch[predict_0,:].cpu().numpy()
                    c_0 = c_batch[predict_0,:].cpu().numpy()
                    l_0 = label_batch[predict_0,:]
                    ee_0 = ee_batch[predict_0,:].cpu().numpy()

                    #
                    batch_num_inCluster[i] = x_0.shape[0] # cluster 안의 포즈 개수
                
            # batch_에 대해 각 cluster의 개수를 넣는다. 
            total_num_inCluster = total_num_inCluster + batch_num_inCluster

            # batch 에 대해 각 cluster 안의 값들이 어떤 label 을 가지는지 저장한다.
            l_o_p = np.zeros(num_priority_label)
            for k in range(num_priority_label):
                l_o_p[k] = (l_0[:,0] == k).nonzero(as_tuple=False)[:,0].cpu().numpy().shape[0]
            total_maxPriority_inCluster[i,:] = total_maxPriority_inCluster[i,:]+l_o_p
        # 해당 cluster 에 해당하는 포즈들을 concate 한다.


    def get_cluster_performance_train_ROT(self, graph, gumbel_temp=1.0, step=0):
        print("get cluster performance")
        progress = tqdm(self.data_loader)
        num_total_data = self.train_dataset.__len__()
        total_batches = len(self.data_loader)

        batch_ind = np.random.randint(0,30+1)
        total_poseValue_inCluster = np.zeros(graph.num_classes)
        total_apdValue_inCluster = np.zeros(graph.num_classes)
        total_num_inCluster = np.zeros(graph.num_classes)
        total_num_mean_prob = 0

        num_priority_label = 3
        total_maxPriority_inCluster = np.zeros((graph.num_classes, num_priority_label))
        
        n_datas = torch.zeros((graph.num_classes))       
        for i_batch, batch in enumerate(progress):

            for k in batch:
                batch[k] = batch[k].to(self.data_device)

            x = batch["x"]          
            cond = batch["cond"]
            ee_cond = batch["ee_cond"]
            ee_gt = batch["ee_gt"]
            label = batch["label"]
            descriptor = batch["descriptor"]
            # condition encoder -> prior decoding
            with torch.no_grad():
                graph.eval()
                nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach().to(self.data_device)

                out_net  = graph.network(des_enc,gumbel_temp,0.0)
                prob = out_net['prob_cat'] # (B*Timestep, 10)

                
                # for accuracy test "load 확인용"
                _, predicted_labels = torch.max(out_net['logits'], dim=1) # (B*T)
                #
                num_cls = prob.shape[-1]
                mean_probab = 1/num_cls
                #
                max_prob,_ = torch.max(prob,1)
                mean_prob_batch = max_prob >= (mean_probab - 0.1)
                mean_prob_batch = (mean_prob_batch <=(mean_probab + 0.1)).nonzero(as_tuple=False)[:,0]
                num_mean_prob = mean_prob_batch.shape[0]
                #
                poseValue_inCluster = np.zeros(num_cls)
                apdValue_inCluster = np.zeros(num_cls)
                for i in range(num_cls):
                    predict_0 = (predicted_labels == i).nonzero(as_tuple=False)[:,0]
                    #
                    x_batch = x.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach() #(B*T,nFeats)
                    c_batch = cond.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach() #(B*T, nFeats)
                    label_batch = label.reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach()
                    ee_batch    = ee_gt.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach()

                    x_0 = x_batch[predict_0,:].cpu().numpy()
                    c_0 = c_batch[predict_0,:].cpu().numpy()
                    l_0 = label_batch[predict_0,:]
                    ee_0 = ee_batch[predict_0,:].cpu().numpy()

                    #
                    poseValue_inCluster[i] = x_0.shape[0] # cluster 안의 포즈 개수
                    #202007
                    #n_datas[i] = n_datas[i] + x_0.shape[0]
                    if x_0.shape[0] > 10:
                        c_0 = c_0[:,:3] # except environment
                        #c_0 = c_0[:,-2643:-2640] # except environment 202207
                        
                        #apd_score = Experiment_utils.get_APD_score_batch(x_0,c_0,1.0,self.data.scaler)
                        #apdValue_inCluster[i] = apd_score # apd score inside cluster
                        
                        n_datas[i] = n_datas[i] + 1
                        

                        #
                        x_0_clip = x_0[np.newaxis,...].copy()
                        c_0_clip = c_0[np.newaxis,...].copy()
                        ee_0_clip = ee_0[np.newaxis,...].copy()
                        
                        if i_batch == batch_ind:
                            self.data.save_animation_UnityFile(x_0_clip,ee_0_clip,c_0_clip, os.path.join(self.log_dir, f'Sample_poses_{i_batch}_{step//1000}k_{i}'))
                    else:
                        apdValue_inCluster[i] = 0
                    #
                    l_o_p = np.zeros(num_priority_label)
                    for k in range(num_priority_label):
                        l_o_p[k] = (l_0[:,0] == k).nonzero(as_tuple=False)[:,0].cpu().numpy().shape[0]
                    total_maxPriority_inCluster[i,:] = total_maxPriority_inCluster[i,:]+l_o_p
                    

                # calc Score ( cluster 안의 APD(low), Cluster안의 포즈 개수( 개수 / 전체 개수, low), prob(1/num_cls) 와 비슷한 값들 개수(low))
                num_inCluster = poseValue_inCluster.copy()
                poseValue_inCluster = poseValue_inCluster/(nBatch_cond*nTimesteps_cond)
                
                # print score
                total_num_mean_prob = total_num_mean_prob + num_mean_prob
                total_poseValue_inCluster = total_poseValue_inCluster + poseValue_inCluster
                total_apdValue_inCluster = total_apdValue_inCluster + apdValue_inCluster
                total_num_inCluster = total_num_inCluster + num_inCluster
                
                #print(f"dist_mean_{num_mean_prob}_poseInSide_{poseValue_inCluster}_apdValue_{apdValue_inCluster}")
        
        for c in range(graph.num_classes):
            total_poseValue_inCluster[c] = total_poseValue_inCluster[c] / n_datas[c]
            total_apdValue_inCluster[c] = total_apdValue_inCluster[c] /n_datas[c]

        # print(f"poseInSideRatio_{(total_poseValue_inCluster)}/{np.sum(total_num_inCluster)}")
        #print(f"apdValue_{(total_apdValue_inCluster)}")

        #print(f"dist_mean_{(total_num_mean_prob)}/{mean_probab:.3f}")
        #print(f"poseInsideNum_{total_num_inCluster}/{np.sum(total_num_inCluster)}")
        print(f"poseInSideRatio_{(total_num_inCluster/(np.sum(total_num_inCluster)))}/{np.sum(total_num_inCluster)}")
        print(f"high_priority_label_{total_maxPriority_inCluster}")

    def get_cluster_performance_train(self, graph, gumbel_temp=1.0, step=0):
        print("get cluster performance")
        progress = tqdm(self.data_loader)
        num_total_data = self.train_dataset.__len__()
    
        total_poseValue_inCluster = np.zeros(graph.num_classes)
        total_apdValue_inCluster = np.zeros(graph.num_classes)
        total_num_inCluster = np.zeros(graph.num_classes)
        total_num_mean_prob = 0

        num_priority_label = 3
        total_maxPriority_inCluster = np.zeros((graph.num_classes, num_priority_label))
        
        n_datas = torch.zeros((graph.num_classes)) 

        batch_ind = 5    
        for i_batch, batch in enumerate(progress):

            for k in batch:
                batch[k] = batch[k].to(self.data_device)

            x = batch["x"]          
            cond = batch["cond"]
            ee_cond = batch["ee_gt"]
            descriptor = batch["sf"]
            # condition encoder -> prior decoding
            with torch.no_grad():
                graph.eval()
                nBatch_cond, nFeatures_cond, nTimesteps_cond = descriptor.shape
                des_enc = descriptor.permute(0,2,1).reshape(-1,nFeatures_cond).clone().detach().to(self.data_device)

                out_net  = graph.network(des_enc,gumbel_temp,0.0)
                prob = out_net['prob_cat'] # (B*Timestep, 10)

                
                # for accuracy test "load 확인용"
                _, predicted_labels = torch.max(out_net['logits'], dim=1) # (B*T)
                #
                num_cls = prob.shape[-1]
                mean_probab = 1/num_cls
                #
                max_prob,_ = torch.max(prob,1)
                mean_prob_batch = max_prob >= (mean_probab - 0.1)
                mean_prob_batch = (mean_prob_batch <=(mean_probab + 0.1)).nonzero(as_tuple=False)[:,0]
                num_mean_prob = mean_prob_batch.shape[0]
                #
                poseValue_inCluster = np.zeros(num_cls)
                apdValue_inCluster = np.zeros(num_cls)
                for i in range(num_cls):
                    predict_0 = (predicted_labels == i).nonzero(as_tuple=False)[:,0]
                    #
                    x_batch = x.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach() #(B*T,nFeats)
                    c_batch = cond.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach() #(B*T, nFeats)
                    ee_batch    = ee_cond.permute(0,2,1).reshape(nBatch_cond*nTimesteps_cond,-1).clone().detach()

                    x_0 = x_batch[predict_0,:].cpu().numpy()
                    c_0 = c_batch[predict_0,:].cpu().numpy()
                    ee_0 = ee_batch[predict_0,:].cpu().numpy()

                    #
                    poseValue_inCluster[i] = x_0.shape[0] # cluster 안의 포즈 개수
                    #
                    if x_0.shape[0] > 10:
                        c_0 = c_0[:,:3] # except environment
                        #apd_score = Experiment_utils.get_APD_score_batch(x_0,c_0,1.0,self.data.scaler)
                        #apdValue_inCluster[i] = apd_score # apd score inside cluster
                        n_datas[i] = n_datas[i] + 1

                        #
                        x_0_clip = x_0[np.newaxis,...].copy()
                        c_0_clip = c_0[np.newaxis,...].copy()
                        ee_0_clip = ee_0[np.newaxis,...].copy()
                        ee_0_clip_r = np.zeros_like(ee_0_clip).copy()
                        if i_batch == batch_ind:
                            self.data.save_animation_withRef(np.concatenate((ee_0_clip, c_0_clip),axis=-1),x_0_clip,x_0_clip, os.path.join(self.log_dir, f'Sample_poses_{i_batch}_{step//1000}k_{i}'))
                    else:
                        apdValue_inCluster[i] = 0
                    

                # calc Score ( cluster 안의 APD(low), Cluster안의 포즈 개수( 개수 / 전체 개수, low), prob(1/num_cls) 와 비슷한 값들 개수(low))
                num_inCluster = poseValue_inCluster
                poseValue_inCluster = poseValue_inCluster/(nBatch_cond*nTimesteps_cond)
                
                # print score
                total_num_mean_prob = total_num_mean_prob + num_mean_prob
                total_poseValue_inCluster = total_poseValue_inCluster + poseValue_inCluster
                total_apdValue_inCluster = total_apdValue_inCluster + apdValue_inCluster
                total_num_inCluster = total_num_inCluster + num_inCluster
                
                #print(f"dist_mean_{num_mean_prob}_poseInSide_{poseValue_inCluster}_apdValue_{apdValue_inCluster}")
        
        for c in range(graph.num_classes):
            total_poseValue_inCluster[c] = total_poseValue_inCluster[c] / n_datas[c]
            total_apdValue_inCluster[c] = total_apdValue_inCluster[c] /n_datas[c]

        # print(f"poseInSideRatio_{(total_poseValue_inCluster)}/{np.sum(total_num_inCluster)}")
        # print(f"apdValue_{(total_apdValue_inCluster)}")

        # print(f"dist_mean_{(total_num_mean_prob)}/{mean_probab:.3f}")
        # #print(f"poseInsideNum_{total_num_inCluster}/{np.sum(total_num_inCluster)}")
        # print(f"high_priority_label_{total_maxPriority_inCluster}")

        print(f"poseInSideRatio_{(total_num_inCluster/(np.sum(total_num_inCluster)))}/{np.sum(total_num_inCluster)}")
        
        
        
