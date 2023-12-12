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
from finetune_tinyautoencoder import TAESD, Clamp, Block, conv, Encoder, Decoder
from src.utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object
from scipy.ndimage import distance_transform_edt
from src.mae_all_data_load import plot_vel_field, GiveMe_loaders, load_vel, VelocityDataset
import gc

class ExtractRep:
    def __init__(self, traj_datset, tae):
        self.traj_dataset = traj_datset
        # self.tae = tae

    def load_velocity(self, flow_dir):
        scl = 1
        # flow_dir = Path(flow_dir)
        # flow_dir = flow_dir.parent
        # flow_dir = flow_dir.parent
        # flow_dir = str(flow_dir)
        all_u_mat = (np.load(flow_dir +'/all_u_mat.npy')*scl)
        all_ui_mat = (np.load(flow_dir +'/all_ui_mat.npy')*scl)
        all_v_mat = (np.load(flow_dir +'/all_v_mat.npy' )*scl)
        all_vi_mat = (np.load(flow_dir +'/all_vi_mat.npy')*scl)
        all_Yi = (np.load(flow_dir +'/all_Yi.npy' )*scl)
        obstacle_mask = (np.load(flow_dir + '/obstacle_mask.npy')*scl)
        obstacle_mask = distance_transform_edt(1-obstacle_mask) # Not sure
        vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi, obstacle_mask]
        return vel_field_data

    def extract_velocity(self, vel_field_data, t, rzn):
        nmodes =  vel_field_data[2].shape[1]
        vx = vel_field_data[0][t,:,:]
        vy = vel_field_data[1][t,:,:] 
        obstacle_mask = vel_field_data[5][t,:,:] 
        obstacle_mask = np.float32(obstacle_mask)

        for m in range(nmodes):
            vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
            vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

        return np.stack([vx, vy, obstacle_mask], axis=0)

    def preprocessing_for_tae(self, vx_vy_list):
        vx_vy_tensor = torch.tensor(vx_vy_list)
        vx_vy_tensor = F.interpolate(vx_vy_tensor, size=(128, 128), mode='bilinear', align_corners=False)     #120,3,512,512
        return vx_vy_tensor    
    
    def extract_latent_rep(self,tae, flow_dir, rzn, return_type='cpu'):
        # device = torch.device("cuda" if torch.cuda.is_available() and return_type=='gpu' else "cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # self.tae.to(device)
        vx_vy_list = []
        vel_data = self.load_velocity(flow_dir) # Shape (list) : [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
        for t in range(vel_data[2].shape[0]): # Extract velocity over all timesteps
            temp = self.extract_velocity(vel_data, t, rzn)
            vx_vy_list.append(temp)
        vx_vy_array = np.array(vx_vy_list) # Shape (array) : (120, 3, 100, 100) 
        preprocessed_vx_vy = self.preprocessing_for_tae(vx_vy_array) # torch.Size([120, 3, 128, 128]) (tensor)

        tae.eval()
        with torch.no_grad():
            outputs_encoder = tae.return_only_enc(preprocessed_vx_vy)  #.to(device))
            latent_reps = outputs_encoder.view(120, -1) # Take mean across second dimension and flatten (120, 4, 16, 16) -> (120, 1024)
            # print(latent_reps[4][:,0].shape) # Verified using self.mae.vit.embeddings(preprocessed_vx_vy)[0][0,0,0:10] and latent_reps[0][0,0,0:10]
    
        if return_type =='cpu':
            return latent_reps.cpu() #shape (120, rep_dim) 1024
        else:
            return latent_reps
        
        
    

if __name__ == "__main__":
    # ROOT = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GenHW_all/"
    # ROOT = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_scaled_0.7_multi_ran_stat_new/"
    ROOT = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_scaled_0.7_multi_ran_stat_new/"
    gather_dir_name = "1_100_2L_withr"
    gather_name = "1_100_2L_withr"
    gather_dir = os.path.join(ROOT, f"Gathered_datasets/gathered_{gather_dir_name}")
    traj_dataset = load_pkl(os.path.join(gather_dir, f"gathered_{gather_name}.pkl"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # tae = TAESD()
    model_name = 'tiny_autoencoder'
    tae_model_name = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Nov23_100envs_clr_randomr/finetuned_tinyautoenc_500D_100E/arch.pt"
    tae = torch.load(tae_model_name).to(device)
    model_state_dict_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Nov23_100envs_clr_randomr/finetuned_tinyautoenc_500D_100E/best_vloss.pt"
    tae.load_state_dict(torch.load(model_state_dict_path)['model_state_dict'])
    
    # tae.to(device)
    
    extract_rep = ExtractRep(traj_dataset, tae)
    save_dir = os.path.join(ROOT, f"Gathered_datasets/gathered_{gather_dir_name}")
    save_name = os.path.join(save_dir, f"gathered_rep_{gather_name}_{model_name}.pkl")   
    # save_interval = len(traj_dataset)/10
    save_interval = 2500

    print(len(traj_dataset))
    for idx in range(len(traj_dataset)):
        _, _, _, _, _,_, _, _, _, _, flow_dir, rzn = traj_dataset[idx] 
        gc.collect()
        Yi_r = extract_rep.extract_latent_rep(tae, flow_dir,rzn, return_type='cpu')
        print(idx)
        print(Yi_r.shape)
        traj_dataset[idx][0] = Yi_r
        print()
        # break
        if idx % save_interval ==0:
            try:
                print(f" saving data at {idx=}")
                save_object(traj_dataset, os.path.join(save_dir, f"gathered_rep_{gather_name}_{model_name}_n{idx}.pkl")  )
            except:
                print(f"save didnt work at {idx=}")

    save_object(traj_dataset, save_name)

    # os.makedirs(save_name, exist_ok=True)
    print()

