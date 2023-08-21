import torch
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import torch.nn.functional as F
import os
# from root_path import ROOT
from pathlib import Path
import sys
from src.utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object
# from train_mae import MAE, ViT, Transformer, PreNorm, FeedForward, Attention
from src.Class_VitMae_MOD import VitMae_model
# from train_timm_mae import 

# from home.rohit.Documents.Research.Planning_with_transformers.Translation_transformer.my-translat-transformer.cfg.config import config

class ExtractRep:
    def __init__(self, traj_datset, mae):
        self.traj_dataset = traj_datset
        self.mae = mae

    def load_velocity(self, flow_dir):
        scl = 1
        flow_dir = Path(flow_dir)
        flow_dir = flow_dir.parent
        flow_dir = flow_dir.parent
        flow_dir = str(flow_dir)
        all_u_mat = (np.load(flow_dir +'/all_u_mat.npy')*scl)
        all_ui_mat = (np.load(flow_dir +'/all_ui_mat.npy')*scl)
        all_v_mat = (np.load(flow_dir +'/all_v_mat.npy' )*scl)
        all_vi_mat = (np.load(flow_dir +'/all_vi_mat.npy')*scl)
        all_Yi = (np.load(flow_dir +'/all_Yi.npy' )*scl)
        vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
        return vel_field_data

    def extract_velocity(self, vel_field_data, t, rzn):
        nmodes =  vel_field_data[2].shape[1]
        vx = vel_field_data[0][t,:,:]
        vy = vel_field_data[1][t,:,:] 

        vx1= 0
        vy1= 0
        vx1= vx1 + vx
        vy1= vy1 + vy

        for m in range(nmodes):
            vx1 += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
            vy1 += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

        return np.stack([vx1,vy1], axis=0)

    def preprocessing_for_mae(self, vx_vy_list):
        vx_vy_tensor = torch.tensor(vx_vy_list)
        vx_vy_tensor = F.interpolate(vx_vy_tensor, size=(224, 224), mode='bilinear', align_corners=False)     #120,2,256,256
        return vx_vy_tensor    
    
    def extract_latent_rep(self, flow_dir, rzn):
        # m = re.search(r'\w+_\w+_\w+_\w+_\w+_\w+_\w+_\w+_\w+', flow_dir)
        # flow_dir_new = flow_dir[:m.end()]
        # dummy_dir = flow_dir_new
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        vx_vy_list = []
        vel_data = self.load_velocity(flow_dir) # Shape (list) : [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
        for t in range(vel_data[2].shape[0]): # Extract velocity over all timesteps
            temp = self.extract_velocity(vel_data, t, rzn)
            vx_vy_list.append(temp)
        vx_vy_list = np.array(vx_vy_list) # Shape (array) : (120, 2, 100, 100) 
        preprocessed_vx_vy = self.preprocessing_for_mae(vx_vy_list) # torch.Size([120, 2, 256, 256]) (tensor)
        self.mae.eval()
        with torch.no_grad():
            #loss = self.mae(preprocessed_vx_vy_list.to(device))
            # latent_reps = self.mae(preprocessed_vx_vy.to(device),loss=False)
            # shape = latent_reps.shape
            # latent_reps = torch.reshape(latent_reps, (shape[0],-1))
            outputs = self.mae(preprocessed_vx_vy.to(device))
            latent_reps = outputs.hidden_states
            print(latent_reps[4][:,0].shape) # Verified using self.mae.vit.embeddings(preprocessed_vx_vy)[0][0,0,0:10] and latent_reps[0][0,0,0:10]
            
        return latent_reps[4][:,0].cpu() #shape (120, rep_dim) 
    

if __name__ == "__main__":
    ROOT = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GenHW_all/"
    gather_dir = "11_500"
    gather_name = 11
    gather_dir = os.path.join(ROOT, f"Gathered_datasets/gathered_{gather_dir}")
    traj_dataset = load_pkl(os.path.join(gather_dir, f"gathered_{gather_name}.pkl"))
    cfg_name = '/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/cfg/contGrid_v5_GenHW.yaml'
    cfg = read_cfg_file(cfg_name=cfg_name)
    mae_model_name = cfg['mae_model_name']

    mae = torch.load(mae_model_name)
    model_name = mae_model_name[:-3].split('/')[-1]
    
    extract_rep = ExtractRep(traj_dataset, mae)
    
    for idx in range(len(traj_dataset)):
        _, _, _, _, _, _, _, _, _, flow_dir, rzn = traj_dataset[idx] 
        Yi_r = extract_rep.extract_latent_rep(flow_dir,rzn)
        print(idx)
        traj_dataset[idx][0] = Yi_r
        # break
    save_dir = os.path.join(ROOT, f"Gathered_datasets/gathered_{gather_name}_500")
    save_name = os.path.join(save_dir, f"gathered_rep_{gather_name}_{model_name}.pkl")
    # os.makedirs(save_name, exist_ok=True)
    save_object(traj_dataset, save_name)
    print()

