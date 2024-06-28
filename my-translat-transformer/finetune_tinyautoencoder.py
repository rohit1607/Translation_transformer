import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import numpy as np
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from src.mae_all_data_load import get_path_names, scale_velocity, load_vel_DG3, load_vel_perlin, VelocityDataset, plot_vel_field, GiveMe_loaders
from src.get_data_names import get_names
from src.Class_optimsAndScheds import Optims_Scheds
from src.utils import make_dir, checkpoint_model, plot_grad_flow
# from src.src_utils import checkpoint_model
from cfg.config_tae import config as sr_config
# from cfg.config_tae import hpt_config
import wandb
from datetime import datetime
import argparse
ARGS_MODE=None

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
        self.max_v = 3.
        self.min_v = -3.
        self.min_sd = 0
        self.max_sd = 10

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
        # x = x.squeeze(dim=0)
        image_raw = self.scale_input(x.to(self.dev))
        image_enc = self.encoder(image_raw.to(dtype=torch.float32)).to(self.dev)
        # image_dec = self.decoder(image_enc).clamp(0, 1)
        # image_dec = self.unscale_output(self.decoder(image_enc))
        image_dec = self.decoder(image_enc)
        return image_dec
    
    def return_only_enc(self, x):
        image_raw = self.scale_input(x.to(self.dev))
        image_enc = self.encoder((image_raw).to(torch.float32))
        return image_enc
    

def train_tae(tae, train_dataloader, val_dataloader, data_sets, eps=5, nth=0, path="", optimizer=0, scheduler=0, start_eps=0, BestValLoss=10e10, BestEpoch=-9):

    # start_time = datetime.now().replace(microsecond=0)
    # start_time_str = start_time.strftime("%m-%d-%H-%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optimizer
    scheduler = None

    tae.to(device)
    print("FineTuning the TAESD model>")
    if device.type == "cuda":
        print("Using CUDA")
    else:
        print("Not using CUDA")
    
    model_save_dir = os.path.join(path, "finetuned_tinyautoenc_"+str(len(data_sets))+"D_"+str(eps)+"E")
    make_dir(model_save_dir)
    checkpoint_model(tae, model_save_dir, None, optimizer,
        scheduler=scheduler, only_save_states=False)
        
    train_loss_epochwise = []
    val_loss_epochwise = []

    bestValLoss = BestValLoss   # must be min of all 
    bestEpoch = BestEpoch

    for epoch in range(start_eps, eps):
        tae.train()
        TLoss= 0
        N= 0
        for _, batch in enumerate(train_dataloader):
            images = batch.to(device).requires_grad_()
            logits = tae(images).requires_grad_()
            # train_loss = outputs.loss
            train_loss = F.mse_loss(logits,images)
            optimizer.zero_grad()
            train_loss.requires_grad = True
            train_loss.backward()
            # plot_grad_flow(tae.named_parameters(), model_save_dir)
            # nn.utils.clip_grad_norm_(tae.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                N= N+1
                TLoss += train_loss
        # scheduler.step()
        TLoss /= N

        VLoss= 0
        N= 0
        tae.eval()
        for _, batch in enumerate(val_dataloader):
            images = batch.to(device)
            with torch.no_grad():
                logits = tae(images)
                # val_loss = outputs.loss
                val_loss = F.mse_loss(logits,images)
                N= N+1
                VLoss += val_loss
        VLoss /= N

        if bestValLoss>VLoss:    # best val loss computation
            bestValLoss=VLoss
            bestEpoch=epoch
            print("saving current model at: " + model_save_dir)
            checkpoint_model(tae, model_save_dir, epoch, optimizer,
                             scheduler=scheduler,
                             loss=(train_loss, val_loss),
                             save_type='cur_best',)

        train_loss_epochwise.append(TLoss.item())
        val_loss_epochwise.append(VLoss.item())
        lr = optimizer.param_groups[0]["lr"]

        print(f"epoch: {epoch}    Train Loss: {TLoss.item()}    Validation Loss: {VLoss.item()}    Learning Rate: {lr}")
        #print(f"{a}   {b}   {c}   {d}")

        logDict= {
            "Epoch": epoch,
            "Train Loss": TLoss.item(),
            "Validation Loss": VLoss.item(),
            "Best Val Loss": bestValLoss,
            "learning rate": lr, #if scheduler.is_available() else lr
            "Best epoch": bestEpoch
        }
        wandb.log(logDict)

        model_dict = {
            "epoch": epoch,
            "model_state_dict": tae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": lr,
            # "scheduler": scheduler.state_dict(),
            "best_val_loss": bestValLoss,
            "best_epoch": bestEpoch,
            "VLoss": VLoss,
            "TLoss": TLoss
        }
        torch.save(model_dict, model_save_dir+"/checkpoints_tinyautoenc_model.pt")

        #for i in range(len(data_sets)):
        if epoch%nth==0 or epoch==eps:
            train_reconstruction_plot(data_sets, tae, 0, sth=0, path=model_save_dir, ep=epoch, st_eps=start_eps)
            train_reconstruction_plot(data_sets, tae, 1, sth=len(data_sets)>>1, path=model_save_dir, ep=epoch, st_eps=start_eps)
            train_reconstruction_plot(data_sets, tae, 2, sth=len(data_sets)-1, path=model_save_dir, ep=epoch, st_eps=start_eps)


    train_loss_epochwise = [loss/(max(train_loss_epochwise)) for loss in train_loss_epochwise]
    val_loss_epochwise = [loss/(max(val_loss_epochwise)) for loss in val_loss_epochwise]

    plt.clf()
    plt.plot([i+1 for i in range(eps-start_eps)], train_loss_epochwise, color='g', label='Train Loss')
    plt.plot([i+1 for i in range(eps-start_eps)], val_loss_epochwise, color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Epochwise')

    plt.savefig(model_save_dir+"/tae_tv_loss_"+str(len(data_sets))+"D_"+str(eps-start_eps)+"E"+".png")

    return train_loss_epochwise, val_loss_epochwise, model_save_dir

def train_reconstruction_plot(dataset, tae, i, sth=0, fname="tae_validation_(", path="", ep=99, st_eps=0):

    taesd = TAESD()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = dataset.__getitem__(sth)
    taesd.to(device)
    
    taesd.eval()
    with torch.no_grad():
        output_image = taesd.unscale_output(taesd(image.unsqueeze(0).to(device)))
     
    image = image.cpu().numpy()    
    output_image = output_image.cpu().numpy()
    # Save at every 5th epoch
    plt.clf()
    tname = fname+str(i)+")dataset_("+str(sth)+")sample"

    if ep==st_eps:
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
        plt.savefig(path+"/"+tname+".png")
        wandb.log({"TRAIN_("+str(i)+")dataset": wandb.Image(path+"/"+tname+".png")})
    
    # im = TF.to_tensor(Image.open(im_path).convert("RGB")).unsqueeze(0).to(dev)

    # # encode image, quantize, and save to file
    # im_enc = taesd.scale_latents(taesd.encoder(im)).mul_(255).round_().byte()
    # enc_path = im_path + ".encoded.png"
    # TF.to_pil_image(im_enc[0]).save(enc_path)
    # print(f"Encoded {im_path} to {enc_path}")

    # # load the saved file, dequantize, and decode
    # im_enc = taesd.unscale_latents(TF.to_tensor(Image.open(enc_path)).unsqueeze(0).to(dev))
    # im_dec = taesd.decoder(im_enc).clamp(0, 1)
    # dec_path = im_path + ".decoded.png"
    # print(f"Decoded {enc_path} to {dec_path}")
    # TF.to_pil_image(im_dec[0]).save(dec_path)

def test_reconstruction_plot(dataset, tae, n_samples=10, fname="tae_test_sample_", path=""):
    taesd = TAESD()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(n_samples):
        r = random.randint(0, len(dataset)-1)
        image = dataset.__getitem__(r)
        
        tae.eval()
        with torch.no_grad():
            output_image = taesd.unscale_output(tae(image.unsqueeze(0).to(device)))
            
        # Save at every 5th epoch
        plt.clf()
        plot_vel_field(output_image[0][0].cpu(), output_image[0][1].cpu(), output_image[0][2].cpu(), flow_name=fname, path=path)
        wandb.log({"TEST_reconstructed_velocity_field_("+str(i)+")dataset": wandb.Image(path+"/"+fname+".png")})
    

@torch.no_grad()
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device", device)
    st = datetime.now().replace(microsecond=0)
    st = st.strftime("%d-%m-%H-%M")
    
    if ARGS_MODE == 'single_run':        # for SINGLE run
        config = sr_config
        exp_name = "GPT_TAE_" + st 
        wandb.init(
            project= "finetune_tae",
            name= exp_name,
            config=config
        )
    else:                                # for SWEEP run
        exp_name = "GPT_TAE_" + st 
        wandb.init(
            name= exp_name,
        )        
    cfg = wandb.config
    
    datapaths = get_names(data_path = cfg.Train_data_path)
    train_dataloader, val_dataloader, data_sets, test_dat = GiveMe_loaders(*datapaths, 
                                       batch_size=cfg.batch_size, path=cfg.tae_path, plot_dat=cfg.plot_dat, cfg=cfg, 
                                       obstacle_mask='data',
                                       dataset='DG3_multi_obs'
                                        #  multi_obs=True
                                         )    
    model = TAESD(encoder_path=None, decoder_path=None).to(device)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

    # train_param
    obj_OS = Optims_Scheds(model=model, optim=cfg.optimizer, sched=cfg.scheduler, cfg=cfg)
    obj_OS.assign()
    optim = obj_OS.optimizer()
    schedul = obj_OS.scheduler()

    train_loss_epochwise, val_loss_epochwise, model_save_dir = train_tae(model, train_dataloader, val_dataloader, data_sets, 
                                                              eps=cfg.epochs, nth=cfg.nth, path=cfg.tae_path,
                                                              optimizer=optim, scheduler=schedul)
    
    # After train, testing on entirely different dataset
    bestModel= torch.load(os.path.join(model_save_dir, "arch.pt")).to(device)
    model_state_dict_path = os.path.join(model_save_dir, "checkpoints_tinyautoenc_model.pt")
    bestModel.load_state_dict(torch.load(model_state_dict_path)['model_state_dict'])
    bestModel.eval()
    
    with torch.no_grad():
        vel_field_data = load_vel_DG3(cfg.aftrain_testDatpath, cfg, obstacle_mask='data')
        dataset = VelocityDataset(vel_field_data, config=cfg , obstacle_mask='data', dataset='DG3_multi_obs')
        plt.clf()
        plot_vel_field(dataset.__getitem__(0)[0], dataset.__getitem__(0)[1], dataset.__getitem__(0)[2], flow_name="aftrain_original", path=model_save_dir)
        wandb.log({"aftrain_original_sample": wandb.Image(model_save_dir+"/aftrain_original.png")})
        test_reconstruction_plot(dataset, bestModel, n_samples=cfg.n_samples, fname="tae_aftrain_test_sample", path=model_save_dir)

        plt.clf()
        plot_vel_field(test_dat.__getitem__(0)[0], test_dat.__getitem__(0)[1],test_dat.__getitem__(0)[2], flow_name="test_original", path=model_save_dir)
        wandb.log({"test_original_sample": wandb.Image(model_save_dir+"/test_original.png")})
        test_reconstruction_plot(test_dat, bestModel, n_samples=cfg.n_samples, fname="tae_test_sample", path=model_save_dir)
    
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    args = parser.parse_args()

    ARGS_MODE = args.mode

    if ARGS_MODE=='single_run':
        main()
    else:
        sweep_id = wandb.sweep(sweep=hpt_config, project="finetune_tae")
        wandb.agent(sweep_id, function=main, count=2)