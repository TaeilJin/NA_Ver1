from sklearn import mixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from . import thops
from . import modules
from . import utils
# modify
from nflows import transforms
from glow.distributions import SSLGaussMixture
import time

def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')
class conditionEncoder_LSTM(nn.Module):
    def __init__(self,in_channels, cond_channels, hidden_channels, out_channels, num_layers):
        super(conditionEncoder_LSTM,self).__init__()
        #self.SAB = SAB(1,1,2)
        self.pose_channels = 66
        self.env_channels = 2640
        self.out_env_channels = 512
        self.in_channel = in_channels

        self.Linear_ENV = nn.Linear(self.env_channels, self.out_env_channels)
        self.Linear_ENV_relu = nn.ReLU()
        self.Linear_ENV_out = nn.Linear(self.out_env_channels, self.out_env_channels)

        cond_channels = cond_channels - self.env_channels + self.out_env_channels
        self.layer_lstm = modules.LSTM(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    
    def forward(self, input):
        #environment
        env = input[:,:,-(self.env_channels+self.pose_channels):-self.pose_channels]
        nBatch, nTimesteps, nFeats = env.shape
        env = env.reshape(-1,nFeats) # (B*T,nFeats)
        env = self.Linear_ENV(env) # (B*T,nDim)
        env = self.Linear_ENV_relu(env)
        env = self.Linear_ENV_out(env)
        env = env.unsqueeze(1).reshape(nBatch,nTimesteps,-1)#(B,Feat,Timesteps)

        input = torch.cat((input[:,:,:-(self.env_channels+ self.pose_channels)],env,input[:,:,-self.pose_channels:]),dim=-1)
        output = self.layer_lstm(input)

        return output

def f(in_channels, out_channels, hidden_channels, cond_channels, network_model, num_layers):
    if network_model=="LSTM":
        return modules.LSTM(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model=="GRU":
        return modules.GRU(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model=="FF":
        return nn.Sequential(
        nn.Linear(in_channels+cond_channels, hidden_channels), nn.ReLU(inplace=False),
        nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=False),
        modules.LinearZeroInit(hidden_channels, out_channels))
    if network_model =="LSTM_ENV":
        return conditionEncoder_LSTM(in_channels, cond_channels, hidden_channels, out_channels, num_layers)

# spline coupling transformation function
coupling_layer_type = 'rational_quadratic_spline'
spline_params = {
    'num_bins': 66,
    'tail_bound': 7.,
    'min_bin_width': 1e-3,
    'min_bin_height': 1e-3,
    'min_derivative': 1e-3,
    'apply_unconditional_transform': False
}

class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine","piecewise"]
    NetworkModel = ["LSTM", "GRU", "FF" , "LSTM_ENV"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels, cond_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 network_model="LSTM",
                 num_layers=2,
                 LU_decomposed=False):
                 
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert network_model in FlowStep.NetworkModel,\
            "network_model should be in `{}`".format(FlowStep.NetworkModel)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels-in_channels // 2, hidden_channels, cond_channels, network_model, num_layers)
        elif flow_coupling == "affine":
            print("affine: in_channels = " + str(in_channels))
            self.f = f(in_channels // 2, 2*(in_channels-in_channels // 2), hidden_channels, cond_channels, network_model, num_layers)
            print("Flowstep pose mask(63) with cond_channels: " + str(cond_channels))
        elif flow_coupling == "piecewise":
            self.transform_fn = transforms.SplitedInputsAwarePiecewiseRationalQuadraticCouplingTransform(
            in_channels=in_channels,
            hidden_channels= hidden_channels,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
            )
            # z1 : half//2 로 parameters 만들고, z2: z- half//2  를 변환시킨다.
            self.half_in_channels = in_channels - (in_channels // 2) 
            outputdim = self.half_in_channels * self.transform_fn._transform_dim_multiplier()
            print("affine: in_channels = " + str(in_channels))
            self.f = f(in_channels // 2, outputdim, hidden_channels, cond_channels, network_model, num_layers)
            print("Flowstep out_channels: " + str(outputdim))
            print("Flowstep pose mask(63) with cond_channels: " + str(cond_channels))
    def init_lstm_hidden(self):
        if self.network_model == "LSTM" or self.network_model == "GRU":
            self.f.init_hidden()
        if self.network_model == "LSTM_ENV":
            self.f.layer_lstm.init_hidden()

    def forward(self, input, cond, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, cond, logdet)
        else:
            return self.reverse_flow(input, cond, logdet)

    def normal_flow(self, input, cond, logdet):
    
        #assert input.size(1) % 2 == 0
        # 1. actnorm
        #z=input
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        z1_cond = torch.cat((z1, cond), dim=1)
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1_cond)
        elif self.flow_coupling == "affine":        
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)+1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2]) + logdet
        elif self.flow_coupling == "piecewise":
            transform_params = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            nBatch3,nSparams3,nTimesteps3 = transform_params.shape
            
            transform_params = transform_params.reshape(nBatch3,self.half_in_channels,-1,nTimesteps3).permute(0,1,3,2)
            
            z1,z2,logdetSp = self.transform_fn.forward(z1,z2,transform_params,context=None)
            
            logdetSp = logdetSp.reshape(nBatch3,-1)
            logdetSp = thops.sum(logdetSp, dim=[1])
            
            logdet = logdetSp + logdet        
        z = thops.cat_feature(z1, z2)
        return z, cond, logdet

    def reverse_flow(self, input, cond, logdet):
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        z1_cond = torch.cat((z1, cond), dim=1)

        #start = time.time()
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = thops.split_feature(h, "cross")
            nan_throw(shift, "shift")
            nan_throw(scale, "scale")
            nan_throw(z2, "z2 unscaled")
            scale = torch.sigmoid(scale + 2.)+1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2]) + logdet
        elif self.flow_coupling == "piecewise":
            transform_params = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            nBatch3,nSparams3,nTimesteps3 = transform_params.shape

            transform_params = transform_params.reshape(nBatch3,self.half_in_channels,-1,nTimesteps3).permute(0,1,3,2)

            z1,z2,logdetSp = self.transform_fn.inverse(z1,z2,transform_params,context=None)
           
            logdetSp = logdetSp.reshape(nBatch3,-1)
            logdetSp = thops.sum(logdetSp, dim=[1])

            logdet = logdetSp + logdet

        #print(" computation flow step time: " , time.time()-start)

        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        nan_throw(z, "z permute_" + str(self.flow_permutation))
       # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, cond, logdet

class SelectLayerUpper(nn.Module):
    def __init__(self):
        super().__init__()
        self.upper_idx = [0,1,2, 
        (1)*3+ 0,(1)*3+ 1,(1)*3+ 2, 
        (2)*3+ 0,(2)*3+ 1,(2)*3+ 2, 
        (3)*3+ 0,(3)*3+ 1,(3)*3+ 2, 
        (4)*3+ 0,(4)*3+ 1,(4)*3+ 2, 
        (5)*3+ 0,(5)*3+ 1,(5)*3+ 2,
        (6)*3+ 0,(6)*3+ 1,(6)*3+ 2, 
        (7)*3+ 0,(7)*3+ 1,(7)*3+ 2, 
        (8)*3+ 0,(8)*3+ 1,(8)*3+ 2, 
        (9)*3+ 0,(9)*3+ 1,(9)*3+ 2,
        (10)*3+0,(10)*3+1,(10)*3+2, 
        (11)*3+0,(11)*3+1,(11)*3+2, 
        (12)*3+0,(12)*3+1,(12)*3+2, 
        (13)*3+0,(13)*3+1,(13)*3+2]
        
        self.dim = len(self.upper_idx)

        # head : 5 left hand: 9 right hand :13
        # vector 의 순서는 head, lh, rh
        self.upper_ee_idx =[
            15,16,17, 
            27,28,29,
            39,40,41]

    def addElement(self, z, z_u ):
        for ii, index in enumerate(self.upper_idx):
            z[:,index,:] = z_u[:,ii,:] 
        return z
    
    def addEndEffectorElement(self,z, ee_cond):
        # 순서가 Head LH, RH 라는 것을 기억하자!
        ee_cond = ee_cond[:,:9,:]
        for ii, index in enumerate(self.upper_ee_idx):
            z[:,index,:] = ee_cond[:,ii,:] 
        return z
    
    def forward(self, input,reverse=False):
        if not reverse:
            self.mask = torch.zeros(input.shape).to(input.device)
            self.mask[:,self.upper_idx,:] = 1
            
            output = input[:,self.upper_idx,:]
            self.masked_pose = input.mul(self.mask)
            
            return output, self.masked_pose
        else:
            self.mask = torch.zeros(input.shape).to(input.device)
            self.mask[:,self.upper_idx,:] = 1

            output = input[:,self.upper_idx,:]
            self.masked_pose = input.mul(self.mask)
            
            return output, self.masked_pose
    
    def getdim(self):
        return self.dim

class SelectLayerLower(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lower_idx = [ 
        (14)*3+0,(14)*3+1,(14)*3+2, (15)*3+0,(15)*3+1,(15)*3+2, (16)*3+0,(16)*3+1,(16)*3+2, (17)*3+0,(17)*3+1,(17)*3+2,
        (18)*3+0,(18)*3+1,(18)*3+2, (19)*3+0,(19)*3+1,(19)*3+2, (20)*3+0,(20)*3+1,(20)*3+2, (21)*3+0,(21)*3+1,(21)*3+2]
        
        self.dim = len(self.lower_idx)
        
        self.lower_ee_idx =[
        (17)*3+0,(17)*3+1,(17)*3+2,
        (21)*3+0,(21)*3+1,(21)*3+2]

    def addElement(self, z, z_l ):
        for ii, index in enumerate(self.lower_idx):
            z[:,index,:] = z_l[:,ii,:] 
        return z
    
    def addEndEffectorElement(self,z, ee_cond):
        # 순서가 HEAD, RH, LH, RF, LF 라는 것을 기억하자!
        ee_cond = ee_cond[:,9:,:]
        for ii, index in enumerate(self.lower_ee_idx):
            z[:,index,:] = ee_cond[:,ii,:] 
        return z
    
    def forward(self, input,reverse=False):
        if not reverse:
            self.mask = torch.zeros(input.shape).to(input.device)
            self.mask[:,self.lower_idx,:] = 1

            output = input[:,self.lower_idx,:]
            self.masked_pose = input.mul(self.mask)
           
            return output, self.masked_pose
        else:
            self.mask = torch.zeros(input.shape).to(input.device)
            self.mask[:,self.lower_idx,:] = 1

            output = input[:,self.lower_idx,:]
            self.masked_pose = input.mul(self.mask)
           
            return output, self.masked_pose
    def getdim(self):
        return self.dim

class FlowNet(nn.Module):
    def __init__(self, x_channels, hidden_channels, cond_channels, K,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 network_model="LSTM",
                 num_layers=2,
                 LU_decomposed=False):
                 
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        N = cond_channels

        # flow upper body selector
        self.select_layer_u = SelectLayerUpper()
        # flow module upper
        self.layers_u = nn.ModuleList()
        for _ in range(K):
            self.layers_u.append(
                FlowStep(in_channels=self.select_layer_u.getdim(),
                         hidden_channels=hidden_channels,
                         cond_channels=N +x_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         network_model=network_model,
                         num_layers=2,
                         LU_decomposed=LU_decomposed))
            self.output_shapes.append(
                [-1, x_channels, 1])

        # flow lower body selector
        self.select_layer_l = SelectLayerLower()
        # flow module lower
        self.layers_l = nn.ModuleList()
        for _ in range(K):
            self.layers_l.append(
                FlowStep(in_channels=self.select_layer_l.getdim(),
                         hidden_channels=hidden_channels,
                         cond_channels=N + x_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         network_model=network_model,
                         num_layers=2,
                         LU_decomposed=LU_decomposed))
            self.output_shapes.append(
                [-1, x_channels, 1])
        
        self.bodycond = torch.zeros((50,66,70))

    def init_lstm_hidden(self):
        for layer in self.layers_u:
            if isinstance(layer, FlowStep):                
                layer.init_lstm_hidden()
        
        for layer in self.layers_l:
            if isinstance(layer, FlowStep):                
                layer.init_lstm_hidden()

    def forward(self, z, cond, ee_cond=None, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            # # train x->z
            # # select upper body first, (fixed hierarcy)
            # z_u = z[:,self.select_layer_u.upper_idx,:]
            # # pose mask condition vector 
            # self.bodycond = self.bodycond.to(z.device)
            # # input End-Effector conditions 
            # if(ee_cond is not None):
            #     self.bodycond = self.select_layer_u.addEndEffectorElement(self.bodycond,ee_cond) # upper body end-effector 를 집어넣는다

            # # flow steps input : x_upper, pose mask condition 
            # z_u_cond = torch.cat((cond,self.bodycond),dim =1)
            # for layer in self.layers_u:
            #     z_u, z_u_cond, logdet = layer(z_u, z_u_cond, logdet, reverse=False)

            # select upper body first, (fixed hierarcy)
            z_u,masked_u = self.select_layer_u(z) 
            # pose mask condition vector 
            masked_init = torch.zeros(masked_u.shape).to(z.device)
            # input End-Effector conditions 
            if(ee_cond is not None):
                masked_init = self.select_layer_u.addEndEffectorElement(masked_init,ee_cond) # upper body end-effector 를 집어넣는다

            # flow steps input : x_upper, pose mask condition 
            z_u_cond = torch.cat((cond,masked_init),dim =1)
            for layer in self.layers_u:
                z_u, z_u_cond, logdet = layer(z_u, z_u_cond, logdet, reverse=False)
            
            # select lower body first, (fixed hierarchy)
            z_l, masked_l = self.select_layer_l(z) 
            # pose mask condition vector 
            # input End-Effector conditions
            if(ee_cond is not None):
                masked_u = self.select_layer_l.addEndEffectorElement(masked_u,ee_cond) # lowerbody end-effector 를 집어넣는다
           
            # flow steps input : x_lower, pose mask condition of upper
            z_l_cond = torch.cat((cond,masked_u),dim =1)
            for layer in self.layers_l:
                z_l, z_l_cond, logdet = layer(z_l, z_l_cond, logdet, reverse=False)
            
            # input new  z & logdet
            z = self.select_layer_u.addElement(z,z_u)
            z = self.select_layer_l.addElement(z,z_l)
            
            # del masked_init
            # del masked_l
            # del masked_u
            return z, logdet
        else:
            # train z->x
            #start = time.time()
            # select upper body first, (fixed hierarcy)
            z_u,masked_u = self.select_layer_u(z) 
            # pose mask condition vector 
            masked_init = torch.zeros(masked_u.shape).to(z.device)
            # input End-Effector positions
            if (ee_cond is not None):
                masked_init = self.select_layer_u.addEndEffectorElement(masked_init,ee_cond) # upper body endeffector positions
            z_u_cond = torch.cat((cond,masked_init), dim =1)
            
            # print(" computation upper select layer time: " , time.time()-start)

            for i,layer in enumerate(reversed(self.layers_u)):
                z_u, z_u_cond, logdet = layer(z_u, z_u_cond, logdet=0, reverse=True)
            
            # !! z_u after reversing all flow steps, is a sampled x_u
            masked_u = self.select_layer_u.addElement(masked_u,z_u)
            #print(" computation upper layer time: " , time.time()-start)

            # input endeffector positions
            if (ee_cond is not None):
                masked_u = self.select_layer_l.addEndEffectorElement(masked_u,ee_cond) # lower body 를 넣어준다.

            # select lower body first, (fixed hierarchy)
            z_l,masked_l = self.select_layer_l(z) 
            # flow steps input : x_lower, pose mask condition of upper
            z_l_cond = torch.cat((cond,masked_u),dim =1)
            for i,layer in enumerate(reversed(self.layers_l)):
                z_l, z_l_cond, logdet = layer(z_l, z_l_cond, logdet=0, reverse=True)
            
            # create x space 
            z = self.select_layer_u.addElement(z,z_u)
            z = self.select_layer_l.addElement(z,z_l)
            
            del masked_init
            del masked_l
            del masked_u

            return z


def get_class_means_latent(net, trainloader, shape, scale=1.):
    ''' use labeled latent representations to compute means '''
    with torch.no_grad():
        means = torch.zeros(shape)
        n_batches = 0
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            n_batches = len(trainloader)
            for (x, x2), y_ in trainloader:
                if len(y_.shape) == 2:
                    y, _ = y_[:, 0], y_[:, 1]
                else:
                    y = y_

                z = net(x)
                for i in range(10):
                    means[i] += z[y == i].reshape((-1,) + means[i].shape).sum(dim=0).cpu()
                    #PAVEL: not the right way of computing means
                progress_bar.set_postfix(max_mean=torch.max(means), 
                                         min_mean=torch.min(means))
                progress_bar.update(x.size(0))

        for i in range(10):
            means[i] /= sum(trainloader.dataset.train_labels[:, 0] == i)

        return means*scale


def get_random_data(net, trainloader, shape, num_means):
    with torch.no_grad():
        x, y = next(iter(trainloader))
        if type(x) in [tuple, list]:
            x = x[0]
        z = net(x)
        idx = np.random.randint(x.shape[0], size=num_means)
        means = z[idx]
        classes = np.unique(y.cpu().numpy())
        for cls in classes:
            if cls == NO_LABEL:
                continue
            means[cls] = z[y==cls][0]
        return means


def get_class_means_data(trainloader, shape, scale=1.):
    ''' use labeled data to compute means '''
    with torch.no_grad():
        means = torch.zeros(shape)
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            for (x, x2), y_ in trainloader:
                if len(y_.shape) == 2:
                    y, _ = y_[:, 0], y_[:, 1]
                else:
                    y = y_

                for i in range(10):
                    means[i] += x[y == i].sum(dim=0).cpu()

        for i in range(10):
            means[i] /= sum(trainloader.dataset.train_labels[:, 0] == i)

        return means*scale


def get_class_means_z(net, trainloader, shape, scale=1.):
    ''' compute latent representation of means in data space '''
    with torch.no_grad():
        means = torch.zeros(shape)
        """
        TODO : initial mean from trained Conditioned Prior
        """
        z_means = net(means)

        return z_means*scale


def get_means(means_type, num_means=10, shape=(3, 32, 32), r=1, trainloader=None, device=None, net=None):

    D = np.prod(shape)
    means = torch.zeros((num_means, D)).to(device)

    if means_type == "from_data":
        print("Computing the means")
        means = get_class_means_data(trainloader, (num_means, *shape), scale=r)
        means = means.reshape((10, -1)).to(device)

    elif means_type == "from_latent":
        print("Computing the means")
        means = get_class_means_latent(net, trainloader, (num_means, *shape), scale=r)
        means = means.reshape((10, -1)).to(device)

    elif means_type == "from_z":
        print("Computing the means")
        means = get_class_means_z(net, trainloader, (num_means, *shape), scale=r)
        means = means.reshape((10, -1)).to(device)

    elif means_type == "pixel_const":
        for i in range(num_means):
            means[i, :] = r * (i-4)
    
    elif means_type == "split_dims":
        mean_portion = D // num_means
        for i in range(num_means):
            means[i, i*mean_portion:(i+1)*mean_portion] = r

    elif means_type == "random":
        for i in range(num_means):
            means[i] = r * torch.randn(D)

    elif means_type == "random_data":
        means = get_random_data(net, trainloader, shape, num_means)

    else:
        raise NotImplementedError(means_type)

    return means



class HierGlow_GMM_ENV(nn.Module):

    def __init__(self, x_channels, cond_channels, hparams):
        super().__init__()
        self.flow = FlowNet(x_channels=x_channels,
                            hidden_channels=hparams.Glow.hidden_channels,
                            cond_channels=cond_channels,
                            K=hparams.Glow.K,
                            actnorm_scale=hparams.Glow.actnorm_scale,
                            flow_permutation=hparams.Glow.flow_permutation,
                            flow_coupling=hparams.Glow.flow_coupling,
                            network_model=hparams.Glow.network_model,
                            num_layers=hparams.Glow.num_layers,
                            LU_decomposed=hparams.Glow.LU_decomposed)
        self.hparams = hparams
        
        # gaussian mixture prior distribution
        self.gaussian_size = hparams.Gumbel.num_classes
        self.x_channels = x_channels
        self.batch_size = hparams.Train.batch_size
        self.means = torch.zeros((self.gaussian_size, x_channels ))
        self.means = get_means("random",num_means=hparams.Gumbel.num_classes, shape=(x_channels), r=1)
        self.vars = torch.ones_like(self.means)
        # gaussian mixture loss fucntion
        if self.means is not None:
            self.loss_fn_GMM = SSLGaussMixture(self.means)
        
        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        assert hparams.Train.batch_size % num_device == 0
        self.z_shape = [hparams.Train.batch_size // num_device, x_channels, 1]
        if hparams.Glow.distribution == "normal":
            self.distribution = modules.GaussianDiag()
        elif hparams.Glow.distribution == "studentT":
            self.distribution = modules.StudentT(hparams.Glow.distribution_param, x_channels)

    def init_lstm_hidden(self):
        self.flow.init_lstm_hidden()

    def forward(self, x=None, cond=None, ee_cond=None, z=None, 
                eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, cond, ee_cond)
        else:
            return self.reverse_flow(z, cond, ee_cond,eps_std)

    def normal_flow(self, x, cond, ee_cond):
    
        n_timesteps = thops.timesteps(x)

        logdet = torch.zeros_like(x[:, 0, 0])

        # encode
        z, objective = self.flow(x, cond, ee_cond, logdet=logdet, reverse=False)

        # prior
        #objective += self.distribution.logp(z)
        # return
        nll = (objective) / float(n_timesteps) #np.log(2.) * 
        return z, nll

    def reverse_flow(self, z, cond, ee_cond, eps_std):
        with torch.no_grad():

            z_shape = self.z_shape
            if z is None:
                z = self.distribution.sample(z_shape, eps_std, device=cond.device)

            x = self.flow(z, cond, ee_cond, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)
    def loss_multiple_gaussian(self,z,cate_prob):
        z= z.permute(0,2,1).reshape(-1,self.x_channels)
       
        mixture_ll = self.loss_fn_GMM.mixture_log_prob(z,cate_prob)
        mixture_ll = mixture_ll.unsqueeze(0)
        mixture_ll = mixture_ll.reshape(self.batch_size,-1)
        mixture_ll = thops.mean(mixture_ll,dim=[-1])
        return mixture_ll
    
