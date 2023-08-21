# File to check how different the representations extracted from MAE are from each other on successive passes
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import random
import torch.nn.functional as F
import os
# from root_path import ROOT
from pathlib import Path
import sys
from src.utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object
from src.train_mae import MAE, ViT, Transformer, PreNorm, FeedForward, Attention
from cfg.config import config
from src.Class_VitMae_MOD import VitMae_model
import matplotlib.pyplot as plt


def load_velocity(flow_dir):
    scl = 1
    all_u_mat = (np.load(flow_dir +'/all_u_mat.npy')*scl)
    all_ui_mat = (np.load(flow_dir +'/all_ui_mat.npy')*scl)
    all_v_mat = (np.load(flow_dir +'/all_v_mat.npy' )*scl)
    all_vi_mat = (np.load(flow_dir +'/all_vi_mat.npy')*scl)
    all_Yi = (np.load(flow_dir +'/all_Yi.npy' )*scl)
    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
    return vel_field_data

def extract_velocity(vel_field_data, t, rzn):
    nmodes = vel_field_data[2].shape[1]
    vx = vel_field_data[0][t,:,:]
    vy = vel_field_data[1][t,:,:] 
    vx1= 0
    vy1= 0
    vx1= vx1 + vx
    vy1= vy1 + vy

    for m in range(nmodes):
        vx1 += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        vy1 += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

    return np.stack([vx1, vy1], axis=0)

class VelocityDataset(Dataset):
    def __init__(self, vel_field_data):
        self.vel_field_data = vel_field_data
    
    def __len__(self):
        return config["nt"] * config["nr"]
    
    def __getitem__(self, idx):
        # rzn = idx // config["nt"]
        rzn = 50
        t = idx % config["nt"]
        # t = 50
        im = extract_velocity(self.vel_field_data, t, rzn)
        
        sz= config["image_size"]   # image size  (256)
        im_tensor = torch.tensor(im)
        # im_tensor = im_tensor.permute(2,0,1)    # 2,100,100
        im_tensor = im_tensor.unsqueeze(0)  # Add batch dimension
        im_tensor = F.interpolate(im_tensor, size=(sz, sz), mode='bilinear', align_corners=False)     # 1,2,256,256
        im_tensor = im_tensor.squeeze(0)  # Remove batch dimension              
        
        return im_tensor

def plot_vel_field(vx_grid, vy_grid, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path=""):
    # Make modes the last axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    xlim, ylim = vx_grid.shape
    Xs = np.arange(0,xlim) + 0.5
    Ys = np.arange(0,ylim) + 0.5
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    im = plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)

    plt.savefig(path+flow_name+".png")
    return im     
    
def extract_latent_rep(dataset, mae, flow_dir, n_samples=1):

    save_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/saved_mae_models/n_samples_reps/1_env/"
    flow_name = flow_dir[:-3].split('/')[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = config["image_size"]
    patch_size = config["patch_size"]
    hw = image_size//patch_size

    for i in range(n_samples):
        r= random.randint(1, len(dataset))
        image_r = dataset.__getitem__(r)
        image_r = torch.reshape(image_r,(1,2,image_size,image_size))
        
        mae.eval()
        with torch.no_grad():
            outputs = mae(image_r.to(device))        
            latent_reps = outputs.hidden_states
            print(latent_reps[-1][:,0].shape) # Verified using self.mae.vit.embeddings(preprocessed_vx_vy)[0][0,0,0:10] and latent_reps[0][0,0,0:10]
            plot_vel_field(image_r[0][0], image_r[0][1], flow_name=flow_name, path=save_path)
            print(f"latent_reps_{flow_name}_{i}")
            np.save(os.path.join(save_path, f"latent_reps_{flow_name}_{i}"), latent_reps[-1][:,0].detach().cpu().numpy())
            

    return None 
    

if __name__ == "__main__":
    ROOT = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GenHW_all/"
    cfg_name = '/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/cfg/contGrid_v5_GenHW.yaml'
    flow_dir_list =[]
    flow_dir_path = f"/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/GenHW/Gathered_datasets/flow_dir_list.txt" 
    with open(flow_dir_path) as f:
        flow_dir_list = [line.rstrip() for line in f] 

    cfg = read_cfg_file(cfg_name=cfg_name)
    mae_model_name = cfg['mae_model_name']
    mae = torch.load(mae_model_name)
    model_name = mae_model_name[:-3].split('/')[-1]

    for flow_dir in flow_dir_list:
        with torch.no_grad():
            vel_field_data = load_velocity(flow_dir)
            dataset = VelocityDataset(vel_field_data)        
            extract_latent_rep(dataset, mae, flow_dir, n_samples=10)


    all_latent_reps=[]

    for flow_dir in flow_dir_list:
        flow_name = flow_dir[:-3].split('/')[-1]
        for i in range(10):
            latent_reps = np.load(os.path.join(f"/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/saved_mae_models/n_samples_reps/1_env", f"latent_reps_{flow_name}_{i}.npy"))
            all_latent_reps.append(latent_reps)
    all_latent_reps = np.stack(all_latent_reps, axis=0)

    A = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            a = all_latent_reps[i]
            b = all_latent_reps[j].T
            A[i][j] = np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

    print(A)
    plt.imshow(A, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig("comparison_of_reps.png")
