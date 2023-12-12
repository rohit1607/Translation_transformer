import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import requests
from io import BytesIO
import numpy as np
# from finetune_tinyautoencoder import TAESD, Clamp, Block, conv, Encoder, Decoder
from scipy.ndimage import distance_transform_edt
import os
import matplotlib.pyplot as plt
from mae_all_data_load import plot_vel_field, plot_vel_field_decoder, plot_vel_imshow
from custom_models import Transformer_causal_encoder

import sys

# dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device", dev)
# taesd = TAESD().to(dev)

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )

def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/taesd_encoder.pth", decoder_path="/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cuda"))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cuda"))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.max_v = 5.
        self.min_v = -5.
        self.min_sd = -10
        self.max_sd = 100

    def scale_input(self,x):
        x[:, :2] = (x[:, :2] - self.min_v) / (self.max_v - self.min_v)  
        x[:, 2] =   (x[:, 2] - self.min_sd) /(self.max_sd - self.min_sd)
        return x
    
    def unscale_output(self,x):
        x[:, :2] = x[:, :2]*(self.max_v - self.min_v)  + self.min_v
        x[:, 2] =  x[:, 2]*(self.max_sd - self.min_sd)  + self.min_sd
        return x
    
    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def forward(self, x):
        # image_raw = TF.to_tensor(x).unsqueeze(0).to(self.dev) #torch.Size([1, 256, 3, 512, 512])
        # image_raw = torch.tensor(x, requires_grad=True).to(self.dev)
        x = x.squeeze(dim=0)
        image_raw = self.scale_input(x.to(self.dev))
        image_enc = self.encoder(image_raw).to(self.dev)
        # image_dec = self.decoder(image_enc).clamp(0, 1)
        # image_dec = self.unscale_output(self.decoder(image_enc))
        image_dec = self.decoder(image_enc)
        return image_dec
    
    def return_only_enc(self, x):
        image_raw = self.scale_input(x.to(self.dev))
        image_enc = self.encoder(image_raw)
        return image_enc

class ExtractRep:
    def __init__(self):
        self.scl = 1

    def load_velocity(self, flow_dir):
        scl = 1
        all_u_mat = (np.load(flow_dir +'/all_u_mat.npy')*scl)
        all_v_mat = (np.load(flow_dir +'/all_v_mat.npy' )*scl)
        # obstacle_mask = (np.load(flow_dir + '/obstacle_mask.npy'))
        # obstacle_mask = distance_transform_edt(1-obstacle_mask) # Not sure
        # np.save(os.path.join(flow_dir, 'signed_dist_obs'), obstacle_mask)
        third_channel = np.zeros((120,100,100))
        # vel_field_data = [all_u_mat, all_v_mat, obstacle_mask]
        vel_field_data = [all_u_mat, all_v_mat, third_channel]
        return vel_field_data

    def extract_velocity(self, vel_field_data, t):
        # nmodes =  vel_field_data[2].shape[1]
        vx = vel_field_data[0][t,:,:]
        vy = vel_field_data[1][t,:,:] 
        obstacle_mask = vel_field_data[2][t,:,:] 
        obstacle_mask = np.float32(obstacle_mask)
        # for m in range(nmodes):
        #     vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        #     vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

        return np.stack([vx, vy, obstacle_mask], axis=0)

    def preprocessing_for_mae(self, vx_vy_list):
        vx_vy_tensor = torch.tensor(vx_vy_list)
        vx_vy_tensor = F.interpolate(vx_vy_tensor, size=(128, 128), mode='bilinear', align_corners=False)     #120,3,512,512
        return vx_vy_tensor    
    
    def extract_latent_rep(self, tae, flow_dir, return_type='cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tae.to(device)
        # self.tae.to(device)
        vx_vy_list = []
        vel_data = self.load_velocity(flow_dir) # Shape (list) : [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
        for t in range(vel_data[2].shape[0]): # Extract velocity over all timesteps
            temp = self.extract_velocity(vel_data, t)
            vx_vy_list.append(temp)
        vx_vy_array = np.array(vx_vy_list) # Shape (array) : (120, 3, 100, 100) 
        preprocessed_vx_vy = self.preprocessing_for_mae(vx_vy_array) # torch.Size([120, 3, 512, 512]) (tensor)
        plot_vel_field(preprocessed_vx_vy[0][0].cpu(), preprocessed_vx_vy[0][1].cpu(), preprocessed_vx_vy[0][2].cpu(), flow_name="tae_test_sample_tgt_onlyv", path=flow_dir)
        output_image = test_reconstruction_plot(preprocessed_vx_vy.to(dtype=torch.float), tae, path=flow_dir)
        output_image_imshow = plot_vel_imshow(output_image[0][0].cpu(), output_image[0][1].cpu(), output_image[0][2].cpu(), flow_name="tae_test_sample_pred_imshow_onlyv", path=flow_dir)
        # tae.to(device)
        tae.eval()
        with torch.no_grad():
            outputs_encoder = tae.return_only_enc(preprocessed_vx_vy.to(dtype=torch.float))  #.to(device))
            latent_reps = outputs_encoder.view(120, -1) # Take mean across second dimension and flatten (120, 4, 16, 16) -> (120, 1024)
            # print(latent_reps[4][:,0].shape) # Verified using self.mae.vit.embeddings(preprocessed_vx_vy)[0][0,0,0:10] and latent_reps[0][0,0,0:10]
    
        if return_type =='cpu':
            return latent_reps.cpu() #shape (120, rep_dim) 1024
        else:
            return latent_reps
    
class ExtractRepP:
    def __init__(self):
        self.scl = 1

    # def load_velocity(self, flow_dir):
    #     scl = 1
    #     all_u_mat = (np.load(flow_dir +'/all_u_mat.npy')*scl)
    #     all_v_mat = (np.load(flow_dir +'/all_v_mat.npy' )*scl)
    #     # obstacle_mask = (np.load(flow_dir + '/obstacle_mask.npy'))
    #     # obstacle_mask = distance_transform_edt(1-obstacle_mask) # Not sure
    #     # np.save(os.path.join(flow_dir, 'signed_dist_obs'), obstacle_mask)
    #     third_channel = np.zeros((120,100,100))
    #     # vel_field_data = [all_u_mat, all_v_mat, obstacle_mask]
    #     vel_field_data = [all_u_mat, all_v_mat, third_channel]
    #     return vel_field_data

    def load_extract_velocity(self):
        scl=1
        vx_vy = (np.load(flow_dir +'/vx_vy.npy')*scl).transpose(1,0,2,3)
        obstacle_mask = np.zeros((120,100,100))
        obstacle_mask = np.float32(obstacle_mask)
        # for m in range(nmodes):
        #     vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        #     vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

        return np.stack([vx_vy[:,0], vx_vy[:,1], obstacle_mask], axis=1)

    def preprocessing_for_mae(self, vx_vy_list):
        vx_vy_tensor = torch.tensor(vx_vy_list)
        vx_vy_tensor = F.interpolate(vx_vy_tensor, size=(128, 128), mode='bilinear', align_corners=False)     #120,3,512,512
        return vx_vy_tensor    
    
    def extract_latent_rep(self, tae, flow_dir, return_type='cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tae.to(device)
        # self.tae.to(device)
        # vel_data = self.load_velocity(flow_dir) # Shape (list) : [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
        # for t in range(vel_data[2].shape[0]): # Extract velocity over all timesteps
        #     temp = self.extract_velocity(vel_data, t)
        #     vx_vy_list.append(temp)
        vx_vy_array = self.load_extract_velocity() # Shape (array) : (120, 3, 100, 100) 
        preprocessed_vx_vy = self.preprocessing_for_mae(vx_vy_array) # torch.Size([120, 3, 512, 512]) (tensor)
        plot_vel_field(preprocessed_vx_vy[0][0].cpu(), preprocessed_vx_vy[0][1].cpu(), preprocessed_vx_vy[0][2].cpu(), flow_name="tae_test_sample_tgt", path=flow_dir)
        output_image = test_reconstruction_plot(preprocessed_vx_vy.to(dtype=torch.float), tae, path=flow_dir)
        output_image_imshow = plot_vel_imshow(output_image[0][0].cpu(), output_image[0][1].cpu(), output_image[0][2].cpu(), flow_name="tae_test_sample_pred_imshow", path=flow_dir)
        # tae.to(device)
        tae.eval()
        with torch.no_grad():
            outputs_encoder = tae.return_only_enc(preprocessed_vx_vy.to(dtype=torch.float))  #.to(device))
            latent_reps = outputs_encoder.view(120, -1) # Take mean across second dimension and flatten (120, 4, 16, 16) -> (120, 1024)
            # print(latent_reps[4][:,0].shape) # Verified using self.mae.vit.embeddings(preprocessed_vx_vy)[0][0,0,0:10] and latent_reps[0][0,0,0:10]
    
        if return_type =='cpu':
            return latent_reps.cpu() #shape (120, rep_dim) 1024
        else:
            return latent_reps    

def test_reconstruction_plot(image, tae, fname="tae_test_sample_pred", path=""):
    taesd = TAESD()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    tae.eval()
    with torch.no_grad():
        output_image = taesd.unscale_output(tae(image.unsqueeze(0).to(device)))
        # Save at every 5th epoch
        plt.clf()
        plot_vel_field(output_image[0][0].cpu(), output_image[0][1].cpu(), output_image[0][2].cpu(), flow_name=fname, path=path)
        return output_image
    
def preprocess_env_reps(env_reps):
    padding_len = None
    traj_len = len(env_reps)
    context_len = 123
    if traj_len > context_len:
        # TODO: correcly write if condition
        sys.exit()
    else:
        padding_len = context_len - traj_len
    env_reps = torch.cat([env_reps, # shape (nenv_reps, 1)
                    # [padding_len+1] is seq_len axis, list(env_reps.shape[1:] is for everythin apart from seq_len axis
                    torch.zeros(([padding_len+1] + list(env_reps.shape[1:])), #[4]+[1]=[4,1]
                    dtype=env_reps.dtype)],
                    dim=0)
    timesteps = torch.arange(start=0, end=context_len, step=1)

    traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                            torch.ones(padding_len, dtype=torch.long)],
                            dim=0).type(torch.bool)
    return timesteps, context_len, env_reps.unsqueeze(dim=0), traj_mask, flow_dir
    

def plot_translate(model: torch.nn.Module, env_reps, earlybreak=10**8):
    model.eval()
    # count = 0           # keeps count of total episodes
    # test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    # set_output = []
    # full_time_mse_list = []
    i_list = [90, 100, 110]
    device = 'cuda'
    # context_len = 123
    # timesteps = 122
    timesteps, context_len, env_reps, traj_mask, flow_dir = preprocess_env_reps(env_reps)
    preds_list = [] # contains tuples of format (flow_dir, rzn, i,preds_post_i(obs_till_i),tgts_post_i )
    with torch.no_grad():
        # for sample in range(len(test_set)):
        # for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in test_dataloader:
        #     if idx%100==0:
        #         print("in translate, idx=", idx)
        #     sample_results = {}

            timesteps = timesteps.to(device)
            # count += 1
            # if count == earlybreak:
            #     break
            src = env_reps[:, :-1, :].to(device)
            tgt = env_reps[:, 1:, :].to(device)
            preds = torch.zeros((1, context_len, tgt.shape[2]),dtype=torch.float32, device=device)
            dyn_inputs = torch.zeros((1, context_len, tgt.shape[2]),dtype=torch.float32, device=device)
            padding_mask = torch.ones((1,context_len)).type(torch.bool).to(device)
 
            loss_post_t = []
            for i in range(1, context_len-1):
                dyn_inputs[0,0:i,:] = src[0,0:i,:] # dyn_inputs start with env data observed till i
                padding_mask[0,:i] = False
                for p in range(i, context_len-1):
                    padding_mask[0,p-1] = False
                    ag_pred_at_p = model(timesteps, dyn_inputs, padding_mask=padding_mask)[0,p-1]
                    dyn_inputs[0,p,:] = ag_pred_at_p
                # IMP_NOTE: dyn_inputs contains observed embs till timestep i (exclusive)
                #                      contains autoregressive predictions from i (inclusive) to context_len-1    
                loss_post_t.append(F.mse_loss(dyn_inputs[0,i:,:],tgt[0,i:,:]))
                if i in i_list:
                    preds_list.append((i, dyn_inputs[0,i:,:], tgt[0,i:,:]))
    return preds_list

def decode_plot(preds_list, log_exp_dir):
    # preds, targets = preds_list[0]
    fname = "env_emb_pred_sample_"
    fname2 = "env_emb_target_sample_"
    taesd = TAESD()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taesd.to(device)
    taesd.eval()
    for j in range(len(preds_list)):
        i, preds, targets = preds_list[j]
        with torch.no_grad():
            preds_image = taesd.unscale_output(taesd.decoder(preds.view(preds.shape[0], 4, 16, 16).to(device)))
            target_image = taesd.unscale_output(taesd.decoder(targets.view(targets.shape[0], 4, 16, 16).to(device)))
        
        samples = np.arange(i, 122, 3) 
        #samples = [100, 103, 106, 109, 112 ..121]
        # i = 100
        # pred_id = [0, 3, .... 21]
        for s in samples:
            r = pred_id = s - i # pred_image and target_image are of len = context_len - obs_len (i.e i)
            plt.clf()
            fig, axs = plt.subplots(1, 2, figsize=(10,5))
            ax = axs[0]
            pred_im = plot_vel_field_decoder(ax,preds_image[r][0].cpu(), preds_image[r][1].cpu(), preds_image[r][2].cpu(), title_name=f"{fname}_t{r}") # flow_name=join(f"figs/i_{i}", fname+f"{r}"), path=log_exp_dir)
            ax = axs[1]
            target_im = plot_vel_field_decoder(ax, target_image[r][0].cpu(), target_image[r][1].cpu(), target_image[r][2].cpu(), title_name=f"{fname2}_t{r}") # flow_name=join(f"figs/i_{i}", fname2+f"{r}"), path=log_exp_dir)
            # ax.set_title(f'env_emb_sample_preds_target_{r}')
            # axs[0].imshow(pred_im)
            # axs[0].set_title(fname+f"{r}")
            # axs[1].imshow(target_im)
            # axs[1].set_title(fname2+f"{r}")
            # plt.tight_layout()
            resulting_image_path = log_exp_dir+"/"+f"real_data_figs/i_{i}/env_emb_sample_{r}"+".png"
            plt.savefig(resulting_image_path)
      
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_path='/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/log/EmbFcast_DG3_model_11-24-11-07.pt'
    envEmb_transformer = torch.load(os.path.join(model_path, 'arch.pt'))
    model_state_dict_path = os.path.join(model_path, 'best_vloss.pt')
    envEmb_transformer.load_state_dict(torch.load(model_state_dict_path)['model_state_dict'])
    # tae_save_dir = '/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Nov23_100envs_clr_randomr/finetuned_tinyautoenc_500D_100E'
    # tae= torch.load(os.path.join(tae_save_dir, "arch.pt")).to(device)
    # tae_state_dict_path = os.path.join(tae_save_dir, "checkpoints_tinyautoenc_model.pt")
    # tae.load_state_dict(torch.load(tae_state_dict_path)['model_state_dict'])
    # tae.eval()
    tae = TAESD()
    
    with torch.no_grad():
        extract_rep = ExtractRepP()
        flow_dir = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/Matlab_data/perlin"
        env_reps = (extract_rep.extract_latent_rep(tae,flow_dir))
        preds_list = plot_translate(envEmb_transformer, env_reps)
        decode_plot(preds_list, model_path)
        print()
    
        
        