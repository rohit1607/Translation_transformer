import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from mae_all_data_load import get_path_names, scale_velocity, load_vel_DG3, load_vel_perlin, VelocityDataset, plot_vel_field, GiveMe_loaders 
from get_data_names import get_names
# from src.Class_optimsAndScheds import Optims_Scheds
# from src.utils import make_dir, checkpoint_model, plot_grad_flow
# from custom_models import TAESD
# from src.src_utils import checkpoint_model
from cfg.config_tae import config as config
import wandb
from datetime import datetime
import argparse
ARGS_MODE=None
from mpl_toolkits.axes_grid1 import make_axes_locatable

wandb.login()

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
        # x[:, 2] =   (x[:, 2] - self.min_sd) /(self.max_sd - self.min_sd)
        return x
    
    def unscale_output(self,x):
        x[:, :2] = x[:, :2]*(self.max_v - self.min_v)  + self.min_v
        # x[:, 2] =  x[:, 2]*(self.max_sd - self.min_sd)  + self.min_sd
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
        # x = x.squeeze(dim=0)
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

def test_reconstruction_plot(image, tae_model, device='cuda', fname="tae_test_sample_pred", path=""):
    tae_model.eval()
    with torch.no_grad():
        output_image = tae_model.unscale_output(tae_model(image.unsqueeze(0).to(device))).cpu().detach().numpy()
        image = image.detach().cpu().numpy()
        
        plt.clf()
        fig, axs = plt.subplots(1, 3, figsize=(15,5), gridspec_kw={'width_ratios': [1, 1, 1]})
        # vmin = min(np.min(output_image[0][0].cpu().numpy()), np.min(image[0].cpu().numpy()))
        # vmax = max(np.max(output_image[0][0].cpu().numpy()), np.max(image[0].cpu().numpy()))
        vmag_image = (image[0]**2 + image[1]**2)**0.5
        vmag_outputimage = (output_image[0][0]**2 + output_image[0][1]**2)**0.5
        vmin = min(np.min(vmag_outputimage), np.min(vmag_image))
        vmax = max(np.max(vmag_outputimage), np.max(vmag_image))        
        # plot_vel_field(dataset.__getitem__(sth)[0], dataset.__getitem__(sth)[1], dataset.__getitem__(sth)[2], flow_name=tname, path=path)
        ax = axs[0]
        original_im = plot_vel_field(ax, vmin, vmax, image[0], image[1], image[2], title_name=f"Original")
        divider = make_axes_locatable(ax)
        cax_vel = divider.append_axes("right", size="5%", pad=0.1)
        cax_vel.axis('off')
        # wandb.log({"TRAIN_original_velocity_field_("+str(i)+")dataset": wandb.Image(path+"/"+tname+".png")}) #flow_name=tname, path=path,
        ax = axs[1]
        # TF.to_pil_image(output_image[0]).save(path+tname+".png")
        # plot_vel_field(output_image[0][0].cpu(), output_image[0][1].cpu(), output_image[0][2].cpu(), flow_name=tname, path=path)
        reconstructed_im = plot_vel_field(ax, vmin, vmax, output_image[0][0], output_image[0][1], output_image[0][2], title_name=f"Reconstructed")
        divider = make_axes_locatable(ax)
        cax_vel = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(reconstructed_im, cax=cax_vel) #, ticks=np.linspace(vmin, vmax, 8)
        cbar.set_label('Velocity')
        # wandb.log({"TRAIN_reconstructed_velocity_field_("+str(i)+")dataset": wandb.Image(path+"/"+tname+".png")})
        ax = axs[2]
        abs_difference = np.abs(image - output_image[0])
        vmag_difference = (abs_difference[0]**2 + abs_difference[1]**2)**0.5
        vmin = np.min(vmag_difference)
        vmax = np.max(vmag_difference)
        mean_error = np.mean(vmag_difference)
        difference_image = plot_vel_field(ax, vmin, vmax, np.abs(image[0] - output_image[0][0]), np.abs(image[1] - output_image[0][1]), np.abs(image[2] - output_image[0][2]), title_name=f"Difference; {mean_error=}")
        divider = make_axes_locatable(ax)
        cax_dif = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(difference_image, cax=cax_dif) #, ticks=np.linspace(vmin, vmax, 8)
        cbar.set_label('Difference')
        # fig.tight_layout()
        plt.savefig(fname)
        wandb.log({"resutls": wandb.Image(fname)})
    


@torch.no_grad()
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device", device)
    st = datetime.now().replace(microsecond=0)
    st = st.strftime("%d-%m-%H-%M")
    
    if ARGS_MODE == 'single_run':        # for SINGLE run
        config = config
        exp_name = "Perlin_TAE_" + st 
        wandb.init(
            project= "test_tae",
            name= exp_name,
            config=config
        )
    else:                                # for SWEEP run
        exp_name = "Perlin_TAE_" + st 
        wandb.init(
            name= exp_name,
        )        
    cfg = wandb.config
    
    datapaths = get_path_names(cfg.Train_data_path, dataset='Perlin')
    train_dataloader, val_dataloader, data_sets, test_dat = GiveMe_loaders(*datapaths, 
                                       batch_size=cfg.batch_size, path=cfg.tae_path, plot_dat=cfg.plot_dat, cfg=cfg)    
    model = TAESD().to(device)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

    # # train_param
    # obj_OS = Optims_Scheds(model=model, optim=cfg.optimizer, sched=cfg.scheduler, cfg=cfg)
    # obj_OS.assign()
    # optim = obj_OS.optimizer()
    # schedul = obj_OS.scheduler()

    # train_loss_epochwise, val_loss_epochwise, model_save_dir = train_tae(model, train_dataloader, val_dataloader, data_sets, 
    #                                                           eps=cfg.epochs, nth=cfg.nth, path=cfg.tae_path,
    #                                                           optimizer=optim, scheduler=schedul)
    
    # # After train, testing on entirely different dataset
    # bestModel= torch.load(os.path.join(model_save_dir, "arch.pt")).to(device)
    # model_state_dict_path = os.path.join(model_save_dir, "checkpoints_tinyautoenc_model.pt")
    # bestModel.load_state_dict(torch.load(model_state_dict_path)['model_state_dict'])
    # bestModel.eval()
    
    model_save_dir = 'Translation_transformer/my-translat-transformer/tmp/debug_tae'
    bestModel =  model

    with torch.no_grad():
        vel_field_data = load_vel_perlin(cfg.aftrain_testDatpath, cfg, obstacle_mask='mag')
        dataset = VelocityDataset(vel_field_data, obstacle_mask='mag', dataset='Perlin')
        i = 0
        image = dataset[i]
        test_reconstruction_plot(image, bestModel, fname="Translation_transformer/my-translat-transformer/tmp/debug_tae/perlin_results.png", path=model_save_dir)

        # test_reconstruction_plot(test_dat, bestModel, fname="tae_test_sample", path=model_save_dir)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    args = parser.parse_args()

    ARGS_MODE = args.mode

    if ARGS_MODE=='single_run':
        main()
    else:
        sweep_id = wandb.sweep(sweep=hpt_config, project="tae_December2023")
        wandb.agent(sweep_id, function=main, count=2)