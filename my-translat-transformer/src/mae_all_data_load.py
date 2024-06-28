"""
loading velocity datasets and sampling time and rzn slices at regular intervals
includes multi gpu datasampling as well


"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import random
import os

nT___ = -1
nR___ = -1
IMSZ = 0


# dataset='DG3', dataset='DG3_multi_obs', dataset='Perlin'
# obstacle_mask =='data', obstacle_mask=='zeros', obstacle_mask=='mag'
def get_path_names(data_path, dataset='DG3_multi_obs'):
    dat_names = []
    if dataset=='DG3':
        for path in os.listdir(data_path):
            if (path == 'nT_120'):
                dat_names.append(data_path+path+"/")
        return dat_names
    elif dataset=='DG3_multi_obs':
        for path in os.listdir(data_path):
            if (path[46:65] == 'multi_ran_stat_new_' and (path[-1]=='0' or path[-1]=='1'or path[-1]=='2' or path[-1]=='3'or path[-1]=='4' or path[-1]=='5' or path[-1]=='6' or path[-1]=='7' or path[-1]=='8' or path[-1]=='9')):
                dat_names.append(data_path+path+"/")
        return dat_names
    elif dataset=='Perlin':
        for path in os.listdir(data_path):
            if (path[0:6]=='Perlin'):
                dat_names.append(data_path+path+'/')
        return dat_names       

def scale_velocity(u, v, F, vmax_by_F):
    speed = np.sqrt(u**2 + v**2)
    max_speed = np.max(speed)
    if max_speed > 0:
        scale_factor= F * vmax_by_F / max_speed
    else:
        scale_factor = 1        
    return scale_factor

def load_vel_DG3(data_path, config, obstacle_mask='data', dataset='DG3_multi_obs'):
    nt=config.nt
    nr=config.nr
    all_u_mat = np.load(data_path +'all_u_mat.npy')
    all_ui_mat = (np.load(data_path +'all_ui_mat.npy'))
    all_v_mat = (np.load(data_path +'all_v_mat.npy'))
    all_vi_mat = (np.load(data_path +'all_vi_mat.npy'))
    all_Yi = np.load(data_path +'all_Yi.npy')
    tts = all_u_mat.shape[0]  # total time steps
    # a = int(config.frac_1*(tts/config.kf)) # Commented by Shubham : Not needed for GPT
    # b = int(config.frac_2*(tts/config.kf)) # Commented by Shubham : Not needed for GPT
    a = 0
    b = 120
    gt = int((b-a)/nt)  # gaps for time steps slicing
    if gt==0:
        gt=1  
    if dataset=='DG3_multi_obs': 
        all_u_mat = all_u_mat[a:b:gt]
        all_ui_mat = all_ui_mat[a:b:gt]
        all_v_mat = all_v_mat[a:b:gt]
        all_vi_mat = all_vi_mat[a:b:gt]        
        scale_factor = scale_velocity(all_u_mat, all_v_mat, 1.0, 2.0)
        all_u_mat *= scale_factor
        all_ui_mat *= scale_factor
        all_v_mat *= scale_factor
        all_vi_mat *= scale_factor
    
    else:
        scale_factor = scale_velocity(all_u_mat, all_v_mat, 1.0, 2.0)
        all_u_mat *= scale_factor
        all_ui_mat *= scale_factor
        all_v_mat *= scale_factor
        all_vi_mat *= scale_factor
        
    all_u_mat[np.isnan(all_u_mat)] = 0
    all_v_mat[np.isnan(all_v_mat)] = 0
    all_ui_mat[np.isnan(all_ui_mat)] = 0
    all_vi_mat[np.isnan(all_vi_mat)] = 0
    all_Yi[np.isnan(all_Yi)] = 0

    if obstacle_mask=='data':
        obstacle_mask = (np.load(data_path + 'obstacle_mask.npy'))[a:b:gt]
        obstacle_mask = distance_transform_edt(1-obstacle_mask)
    elif obstacle_mask=='zeros' and obstacle_mask=='mag':
        obstacle_mask = np.zeros_like(all_u_mat)

    trzn = all_Yi.shape[1]  # total realizations
    gr = int(trzn/nr)  # gaps for rzn steps slicing
    
    random_indices = np.random.choice(all_Yi.shape[1], 5, replace=False)
    all_Yi = all_Yi[a:b:gt, random_indices, :]

    global nT___, nR___
    nT___ = all_u_mat.shape[0]
    nR___ = all_Yi.shape[1]

    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi, obstacle_mask]

    # global IMSZ
    # IMSZ = config.image_size

    return vel_field_data

def load_vel_perlin(flow_dir, obstacle_mask):
    scl = 1
    vx_vy = np.float32(np.load(flow_dir +'/all_vxvy.npy')*scl).transpose(1,0,2,3)
    if obstacle_mask=='data':
        obstacle_mask = (np.load(flow_dir + 'obstacle_mask.npy')*scl)
        obstacle_mask = distance_transform_edt(1-obstacle_mask)
    elif obstacle_mask=='zeros':
        obstacle_mask = np.zeros((60,50,50))
        obstacle_mask = np.float32(obstacle_mask)
    elif obstacle_mask=='mag':
        obstacle_mask = np.sqrt(vx_vy[:,0]**2 + vx_vy[:,1]**2)
        obstacle_mask = np.float32(obstacle_mask)
    vel_field_data = [vx_vy, obstacle_mask]    
    return vel_field_data

def extract_velocity_DG3(vel_field_data, t, rzn, as_image=True, obstacle_mask='data'):
    #  TODO: hardcoding scale to lie between 0,1. Figure uncsaling oout later
    nmodes =  vel_field_data[2].shape[1]
    vx = vel_field_data[0][t,:,:].copy()
    vy = vel_field_data[1][t,:,:].copy()
    for m in range(nmodes):
        vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]
    if obstacle_mask=='data':
        obstacle_mask = vel_field_data[5][t,:,:] 
        obstacle_mask = np.float32(obstacle_mask)
    elif obstacle_mask=='mag':
        obstacle_mask = np.sqrt(vx**2 + vy**2)
        obstacle_mask = np.float32(obstacle_mask)
    else:
        obstacle_mask = np.float32(np.zeros_like(vx))
    
    # TODO: remove or take care of this later
    # vmag = np.sqrt(vx**2 + vy**2)
    # vx = (vx - np.min(vx))/(np.max(vx) - np.min(vx))
    # vy = (vy - np.min(vy))/(np.max(vy) - np.min(vy))
    # obstacle_mask = obstacle_mask/np.max(obstacle_mask)
    
    if as_image:
        im = np.stack([vx, vy, obstacle_mask], axis=0)
        return im
    else:
        return vx, vy
    
def extract_velocity_perlin(vel_field_data, as_image=True):
    vx_vy = vel_field_data[0]
    obstacle_mask = vel_field_data[1]
    if as_image:
        im = np.stack([vx_vy[:,0], vx_vy[:,1], obstacle_mask], axis=1)
        return im
    else:
        return vx_vy[:,0], vx_vy[:,1]
    
    
class VelocityDataset(Dataset):
    def __init__(self, vel_field_data, nr=-1, nt=-1, config=None, obstacle_mask='data', dataset='DG3_multi_obs', return_stats=False):
        self.vel_field_data = vel_field_data
        global nT___, nR___, IMSZ
        self.obstacle_mask = obstacle_mask 
        self.nt = nT___ if nT___!=-1 else nt
        self.nr = nR___ if nR___!=-1 else nr
        # assert(self.nt!=-1 and self.nr!=-1)
        self.dataset = dataset
        self.config = config
        
    # def get_stats(self):
    #     im = extract_velocity(self.vel_field_data, 0, 0, obstacle_mask=self.obstacle_mask)

    #     return max_v, min_v, max_c3. min_c3
    def __len__(self):
        return self.nt * self.nr
    
    def normalize(self, channel, eps=1e-5):
        cmin, cmax = torch.min(channel), torch.max(channel)
        channel = (channel - cmin) / (cmax - cmin + eps)
        return channel, (cmin,cmax)
        
    def normalize_each_channel(self, im):
        # TODO: take min and max across axis=1
        norm_im = torch.zeros_like(im)      
        norm_im[0], stats_c0 =  self.normalize(im[0])
        norm_im[1], stats_c1 = self.normalize(im[1])
        norm_im[2], stats_c2 = self.normalize(im[2])
        return norm_im, (stats_c0, stats_c1, stats_c2)    
    
    def __getitem__(self, idx):
        rzn = idx // self.nt
        t = idx % self.nt
        sz= self.config.image_size   # image size  (256)
        if self.dataset == 'DG3_multi_obs' or self.dataset=='DG3':
            im = extract_velocity_DG3(self.vel_field_data, t, rzn, obstacle_mask=self.obstacle_mask)
            im_tensor = torch.tensor(im)
            im_tensor = im_tensor.unsqueeze(0)  # Add batch dimension
            im_tensor = F.interpolate(im_tensor, size=(sz, sz), mode='bilinear', align_corners=False)     # 1,3,256,256
            im_tensor = im_tensor.squeeze(0)  # Remove batch dimension 
        else:
            im = extract_velocity_perlin(self.vel_field_data)
            im_tensor = torch.tensor(im)
            im_tensor = F.interpolate(im_tensor, size=(sz, sz), mode='bilinear', align_corners=False)     # 1,3,256,256             
        # im_tensor_ = torch.tensor(im_tensor, requires_grad=True)
        im_tensor, stats = self.normalize_each_channel(im_tensor)
        return im_tensor, stats


    
def GiveMe_loaders(*args, batch_size=16, path="", plot_dat=True, multi_gpu=False, cfg=0, obstacle_mask='data', dataset='DG3_multi_obs'):
    N= len(args)
    data_sets= []
    for i in range(N):
        if dataset=='DG3' or dataset=='DG3_multi_obs':
            vel_field_data_i = load_vel_DG3(args[i], cfg, obstacle_mask=obstacle_mask, dataset=dataset)
            dataset_i = VelocityDataset(vel_field_data_i, config=cfg, obstacle_mask=obstacle_mask, dataset=dataset)
            data_sets.append(dataset_i)
        else:
            vel_field_data_i = load_vel_perlin(args[i], obstacle_mask)
            dataset_i = VelocityDataset(vel_field_data_i, config=cfg, obstacle_mask=obstacle_mask, dataset=dataset)
            data_sets.append(dataset_i)
            
        if plot_dat:
            plot_vel_field(dataset_i.__getitem__(0)[0], dataset_i.__getitem__(0)[1], flow_name="loader_flow_"+str(i), path=path)
        
    dataset = ConcatDataset(data_sets)
    Len= len(dataset)
    assert((Len*(cfg.val_size + cfg.test_size + cfg.train_size) - Len)==0)
    g1 = torch.Generator().manual_seed(cfg.random_seed)
    train_dat, val_dat, test_dat= random_split(dataset, [cfg.train_size, cfg.val_size, cfg.test_size], generator=g1)
    if multi_gpu:
        # shuffle has to be kept false because shuffling is done by Dsampler
        train_dataloader = DataLoader(train_dat, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dat))
        val_dataloader = DataLoader(val_dat, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(val_dat))
    else:
        train_dataloader = DataLoader(train_dat, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dat, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, val_dat, test_dat

# def plot_vel_field(ax, vx_grid, vy_grid, obs_mask, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path="", title_name=""):
#     # Make modes the last axis
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal', adjustable='box')

#     vx_grid = np.flipud(vx_grid)
#     vy_grid = np.flipud(vy_grid)
#     xlim, ylim = vx_grid.shape
#     Xs = np.arange(0,xlim) + 0.5
#     Ys = np.arange(0,ylim) + 0.5
#     X,Y = np.meshgrid(Xs, Ys)
#     ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
#     # plt.imshow(obs_mask, origin='lower', alpha=0.7)
#     v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
#     im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)
#     ax.set_title(title_name)
#     if not (flow_name=="" and path==""):
#         plt.savefig(path+"/"+flow_name+".png")
#     return im 
# def plot_vel_field(vx_grid, vy_grid, obs_mask, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path=""):
#     # Make modes the last axis
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal', adjustable='box')

#     vx_grid = np.flipud(vx_grid)
#     vy_grid = np.flipud(vy_grid)
#     xlim, ylim = vx_grid.shape
#     Xs = np.arange(0,xlim) + 0.5
#     Ys = np.arange(0,ylim) + 0.5
#     X,Y = np.meshgrid(Xs, Ys)
#     plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
#     # plt.imshow(obs_mask, origin='lower', alpha=0.7)
#     v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
#     im = plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)

#     if not (flow_name=="" and path==""):
#         plt.savefig(path+"/"+flow_name+".png")
#     return im 

def plot_vel_field(ax, vmin, vmax, vx_grid, vy_grid, obs_mask, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path="", title_name="",conturf_mat=None):
    # Make modes the last axis
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    xlim, ylim = vx_grid.shape #TODO remove hardcode for perlin
    Xs = np.arange(0,xlim) + 0.5
    Ys = np.arange(0,ylim) + 0.5
    X,Y = np.meshgrid(Xs, Ys)
    ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    # plt.imshow(obs_mask, origin='lower', alpha=0.7)
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    if conturf_mat is None:
        conturf_mat = v_mag_grid
    im = ax.contourf(X, Y, conturf_mat, cmap = "Blues", alpha = 0.9, zorder = -1e5, vmin=vmin, vmax=vmax)
    ax.set_title(title_name)
    if not (flow_name=="" and path==""):
        plt.savefig(path+"/"+flow_name+".png")
    return im   

def plot_vel_field_decoder(ax, vmin, vmax, vx_grid, vy_grid, obs_mask, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path="", title_name=''):
    # Make modes the last axis
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    # Scale values to lie between 0 - 1
    # min_v = -5 # -10 
    # max_v = 5 #10
    # vx_grid = (vx_grid - min_v) / (max_v - min_v)
    # vy_grid = (vy_grid - min_v) / (max_v - min_v)
    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    xlim, ylim = vx_grid.shape
    Xs = np.arange(0,xlim) + 0.5
    Ys = np.arange(0,ylim) + 0.5
    X,Y = np.meshgrid(Xs, Ys)
    ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    # ax.imshow(obs_mask, origin='lower', alpha=0.7)
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    cmap = plt.get_cmap("Blues")
    # new_cmap = plt.cm.colors.ListedColormap(cmap(np.linspace(vmin, vmax, 10)))
    im = ax.contourf(X, Y, v_mag_grid, cmap = 'Blues', alpha = 0.9, zorder = -1e5, vmin=vmin, vmax=vmax)
    ax.set_title(title_name)
    # ax.set_clim(vmin, vmax) 
    if not (flow_name=="" and path==""):
        plt.savefig(path+"/"+flow_name+".png")
    return im  

def plot_vel_imshow(vx_grid, vy_grid, obs_mask, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path=""):
    # Make modes the last axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    # Scale values to lie between 0 - 1
    # min_v = -5 # -10 
    # max_v = 5 #10
    # min_sd = -7
    # max_sd = 89
    # vx_grid = (vx_grid - min_v) / (max_v - min_v)
    # vy_grid = (vy_grid - min_v) / (max_v - min_v)
    # obs_mask = (obs_mask - min_sd) /(max_sd - min_sd)
    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    xlim, ylim = vx_grid.shape
    Xs = np.arange(0,xlim) + 0.5
    Ys = np.arange(0,ylim) + 0.5
    X,Y = np.meshgrid(Xs, Ys)
    # plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    im = plt.imshow(np.stack([vx_grid,vy_grid,obs_mask.numpy()], axis=2))
    # v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    # im = plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)

    if not (flow_name=="" and path==""):
        plt.savefig(path+"/"+flow_name+".png")
    return im   

# if __name__ == "__main__":

#     a = "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/"
#     data_path_1 = a+"GenHW_TV_DNV_dl_c83_m10_s20_f2_A1/"
#     data_path_2 = a+"GenHW_TV_DNV_dl_c-36_m20_s20_f2_A1/"
#     data_path_3 = a+"GenHW_TV_DNV_dl_c-17_m10_s20_f2_A1/"
#     data_path_4 = a+"GenHW_TV_DNV_dl_c0_m0_s20_f2_A1/"
#     data_path_5 = a+"GenHW_TV_DNV_dl_c50_m0_s20_f2_A1/"
#     data_path_6 = a+"GenHW_TV_DNV_dl_c64_m20_s20_f2_A1/"
#     data_path_7 = a+"GenHW_TV_DNV_dl_c14_m20_s20_f2_A1/"
#     data_path_8 = a+"GenHW_TV_DNV_dl_c33_m10_s20_f2_A1/"
#     data_path_9 = a+"GenHW_TV_DNV_dl_c42_m5_s20_f2_A1/"
#     data_path_10 = a+"GenHW_TV_DNV_dl_c54_m25_s20_f2_A1/"

#     loader_path = config["loader_path"]

#     # m= -80, +80, -45, +45, 0
#     # c= 10, 90, 35, 65

#     T_loader, V_loader = GiveMe_loaders(data_path_1, data_path_2, data_path_3, data_path_4, data_path_5, data_path_6, data_path_7, data_path_8, data_path_9, data_path_10,
#                                          val_size=0.05, batch_size=16, path=loader_path)



# END    