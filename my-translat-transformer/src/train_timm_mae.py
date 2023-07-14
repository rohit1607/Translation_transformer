import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

import numpy as np

import random

from einops import rearrange, repeat

from sab_data_load import plot_vel_field, GiveMe_loaders, load_vel, VelocityDataset

from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

from testing import test_reconstruction_plot

import wandb

from datetime import datetime

from cfg.config import hpt_config
from cfg.config import config as sr_config

from get_data_names import get_names

import argparse
ARGS_MODE = None


wandb.login()

model_name = ""  # to save model name


def unmasked_ids(mask_space):
    Len = len(mask_space)
    un_ids = []
    for i in range(Len):
        if mask_space[i]==0:
            un_ids.append(i)
    un_ids = np.array(un_ids)
    un_ids = torch.from_numpy(un_ids)
    un_ids.unsqueeze_(0)
    return un_ids



def test_reconstruction_plot(dataset, mae, n_samples=10, fname="timm_test_sample_", path=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    szImg= wandb.config.image_size
    p12= wandb.config.patch_size  # patch size (our case always remains square so total pixels=p12*p12*c)
    hw= szImg//p12  #number of such (p12 X p12) images along height and width
    
    for i in range(n_samples):
        r= random.randint(1, len(dataset))
        image_r = dataset.__getitem__(r)
        image_r = torch.reshape(image_r,(1,2,szImg,szImg))
        image_patch = rearrange(image_r,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)
        unmasked_img = np.zeros((1, hw*hw, p12*p12*2))


        # as we just need to get inferences out of our model
        mae.eval()
        with torch.no_grad():
            outputs = mae(image_r.to(device))
            re_image_r = outputs.logits  #b.n.p1*p2*c    (1.256.512)
            unmasked_ind = unmasked_ids(outputs.mask[0])


        re_image_r = re_image_r.cpu().detach().numpy()
        ind_list = unmasked_ind[0].cpu().numpy()  


        unmasked_img = np.zeros((1, hw*hw, p12*p12*2))
        for ind in ind_list:
            re_image_r[:,ind,:] = image_patch[:,ind,:]
            unmasked_img[:,ind,:] = image_patch[:,ind,:]


        re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)
        unmasked_img = rearrange(unmasked_img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)    

        re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)

        plt.clf()
        plot_vel_field(re_img[0][0], re_img[0][1], flow_name=fname+str(i), path=path)
        wandb.log({"TEST_reconstructed_velocity_field": wandb.Image(path+fname+str(i)+".png")})



def train_reconstruction_plot(dataset, mae, i, nth=0, fname="timm_Train_flow_(", path="", ep=99):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    szImg= wandb.config.image_size
    p12= wandb.config.patch_size  # patch size (our case always remains square so total pixels=p12*p12*c)
    hw= szImg//p12  #number of such (p12 X p12) images along height and width
    
    image_r = dataset.__getitem__(nth)
    image_r = torch.reshape(image_r, (1, 2, szImg, szImg))
    image_patch = rearrange(image_r,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)
    unmasked_img = np.zeros((1, hw*hw, p12*p12*2))


    # as we just need to get inferences out of our model
    mae.eval()
    with torch.no_grad():
        outputs = mae(image_r.to(device))
        re_image_r = outputs.logits  #b.n.p1*p2*c    (1.256.512)
        unmasked_ind = unmasked_ids(outputs.mask[0])
        # print(f"{re_image_r.shape}      {unmasked_ind.shape}")
        # print(outputs.mask.shape)
        a = outputs.hidden_states
        print(f"{len(a)}     {a[0].shape}      {a[1].shape}      {a[2].shape}     {a[3].shape}     {a[4].shape}")
        #print(a[0])


    re_image_r = re_image_r.cpu().detach().numpy()
    ind_list = unmasked_ind[0].cpu().numpy()  


    unmasked_img = np.zeros((1, hw*hw, p12*p12*2))
    for ind in ind_list:
        re_image_r[:,ind,:] = image_patch[:,ind,:]
        unmasked_img[:,ind,:] = image_patch[:,ind,:]


    re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)
    unmasked_img = rearrange(unmasked_img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)    


    re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 2)
    
    # Save at every 5th epoch
    plt.clf()
    tname = fname+str(i)+")dataset_("+str(nth)+")sample"

    if ep==0:
        plot_vel_field(dataset.__getitem__(nth)[0], dataset.__getitem__(nth)[1], flow_name=tname, path=path)
        wandb.log({"TRAIN_original_velocity_field_("+str(i)+")dataset": wandb.Image(path+tname+".png")})
        
    plot_vel_field(re_img[0][0], re_img[0][1], flow_name=tname, path=path)
    wandb.log({"TRAIN_reconstructed_velocity_field_("+str(i)+")dataset": wandb.Image(path+tname+".png")})






def train_timm_mae(mae, train_dataloader, val_dataloader, data_sets, lr=0.0001, wd=0.0, fc=0.85, pat=3, eps=5, nth=0, path=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=wd)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=fc, patience=pat)

    mae.to(device)
    if device.type == "cuda":
        print("Using CUDA")
    else:
        print("Not using CUDA")
        
    train_loss_epochwise = []
    val_loss_epochwise = []

    bestValLoss = 10e10    # must be min of all 
    bestEpoch = -9

    for epoch in range(eps):
        TLoss= 0
        N= 0
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch.to(device)
            optimizer.zero_grad()
            outputs = mae(images)
            train_loss = outputs.loss
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                TLoss += train_loss
                N= batch.shape[0]
        scheduler.step(train_loss)
        TLoss /= N
        
        VLoss= 0
        N= 0
        for batch_idx, batch in enumerate(val_dataloader):
            images = batch.to(device)
            with torch.no_grad():
                outputs = mae(images)
                val_loss = outputs.loss
                VLoss += val_loss
                N= batch.shape[0]
        VLoss /= N 

        global model_name
        if bestValLoss>VLoss:    # best val loss computation
            bestValLoss=VLoss
            bestEpoch=epoch
            model_name = path+"best_timm_maeModel_"+str(len(data_sets))+"D_"+str(eps)+"E"+".pt"
            torch.save(mae, model_name)

        train_loss_epochwise.append(TLoss.item())
        val_loss_epochwise.append(VLoss.item())

        print(f"epoch: {epoch}    Train Loss: {TLoss.item()}   Validation Loss: {VLoss.item()}")
        #print(f"{a}   {b}   {c}   {d}")

        logDict= {
            "Epoch": epoch,
            "Train Loss": TLoss.item(),
            "Validation Loss": VLoss.item(),
            "Best Val Loss": bestValLoss,
            "learning rate": optimizer. param_groups[0]["lr"], #if scheduler.is_available() else lr
            "Best epoch": bestEpoch
        }
        wandb.log(logDict)

        #for i in range(len(data_sets)):
        train_reconstruction_plot(data_sets[4], mae, 4, nth=nth, path=path, ep=epoch)
        train_reconstruction_plot(data_sets[9], mae, 9, nth=nth, path=path, ep=epoch)



    train_loss_epochwise = [loss/(max(train_loss_epochwise)) for loss in train_loss_epochwise]
    val_loss_epochwise = [loss/(max(val_loss_epochwise)) for loss in val_loss_epochwise]

    plt.clf()
    plt.plot([i+1 for i in range(eps)], train_loss_epochwise, color='g', label='Train Loss')
    plt.plot([i+1 for i in range(eps)], val_loss_epochwise, color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Epochwise')

    plt.savefig(path+"timm_mae_tv_loss_"+str(len(data_sets))+"D_"+str(eps)+"E"+".png")

    return train_loss_epochwise, val_loss_epochwise






def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # a = "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/"
    # data_path_1 = a+"GenHW_TV_DNV_dl_c83_m10_s20_f2_A1/"
    # data_path_2 = a+"GenHW_TV_DNV_dl_c-36_m20_s20_f2_A1/"
    # data_path_3 = a+"GenHW_TV_DNV_dl_c-17_m10_s20_f2_A1/"
    # data_path_4 = a+"GenHW_TV_DNV_dl_c0_m0_s20_f2_A1/"
    # data_path_5 = a+"GenHW_TV_DNV_dl_c50_m0_s20_f2_A1/"
    # data_path_6 = a+"GenHW_TV_DNV_dl_c64_m20_s20_f2_A1/"
    # data_path_7 = a+"GenHW_TV_DNV_dl_c14_m20_s20_f2_A1/"
    # data_path_8 = a+"GenHW_TV_DNV_dl_c33_m10_s20_f2_A1/"
    # data_path_9 = a+"GenHW_TV_DNV_dl_c42_m5_s20_f2_A1/"
    # data_path_10 = a+"GenHW_TV_DNV_dl_c54_m25_s20_f2_A1/"


    st = datetime.now().replace(microsecond=0)
    st = st.strftime("%d-%m %H-%M")

    if ARGS_MODE == 'single_run':        # for SINGLE run
        config = sr_config
        exp_name = "TIMM_single_mae_88m " + st 
        wandb.init(
            project= "s'jar",
            name= exp_name,
            config=config
        )
    else:                                # for SWEEP run
        exp_name = "TIMM_multiple_mae_88m " + st 
        wandb.init(
            name= exp_name,
        )
    
    
    cfg=wandb.config

    # loader
    # train_dataloader, val_dataloader, data_sets = GiveMe_loaders(data_path_1, data_path_2, data_path_3, data_path_4, data_path_5, data_path_6, data_path_7, data_path_8, data_path_9, data_path_10, 
    #                                    val_size=cfg.val_size, batch_size=cfg.batch_size, path=cfg.timm_path)
    
    datpaths = get_names(data_path = cfg.Train_data_path)
    train_dataloader, val_dataloader, data_sets = GiveMe_loaders(*datpaths, 
                                       val_size=cfg.val_size, batch_size=cfg.batch_size, path=cfg.timm_path)


    # Initializing a ViT MAE vit-mae-base style configuration
    # timm_VitMae
    configuration = ViTMAEConfig(
                                hidden_size = cfg.hidden_size,    # encoder_dim
                                num_hidden_layers = cfg.num_hidden_layers,    # encoder_depth
                                num_attention_heads = cfg.num_attention_heads,    # encoder_heads
                                intermediate_size = cfg.intermediate_size,    # encoder_ffn out_dim
                                hidden_act = cfg.hidden_act,    # ffn activation_fxn
                                hidden_dropout_prob = cfg.hidden_dropout_prob,
                                attention_probs_dropout_prob = cfg.attention_probs_dropout_prob,
                                initializer_range = cfg.initializer_range,
                                layer_norm_eps = cfg.layer_norm_eps,
                                image_size = cfg.image_size,
                                patch_size = cfg.patch_size,
                                num_channels = cfg.num_channels,
                                qkv_bias = cfg.qkv_bias,
                                decoder_num_attention_heads = cfg.decoder_num_attention_heads,    # decoder_heads
                                decoder_hidden_size = cfg.decoder_hidden_size,    # decoder_dim
                                decoder_num_hidden_layers = cfg.decoder_num_hidden_layers,    # decoder_depth
                                decoder_intermediate_size = cfg.decoder_intermediate_size,
                                mask_ratio = cfg.mask_ratio,
                                norm_pix_loss = cfg.norm_pix_loss,
                                output_hidden_states = True
                                )

    # Initializing a model (with random weights) from the vit-mae-base style configuration
    model = ViTMAEForPreTraining(configuration)

    # Accessing the model configuration
    configuration = model.config
    #print(configuration)


    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params


    # train_param
    train_loss_epochwise, val_loss_epochwise = train_timm_mae(model, train_dataloader, val_dataloader, data_sets, 
                                                              wd=cfg.wd, lr=cfg.lr, eps=cfg.epochs, fc=cfg.opt_factor, pat=cfg.patience, nth=cfg.nth, path=cfg.timm_path)


    # After train, testing on entirely different dataset
    global model_name
    bestModel= torch.load(model_name).to(device)
    bestModel.eval()
    with torch.no_grad():
        vel_field_data = load_vel(cfg.aftrain_testDatpath)
        dataset = VelocityDataset(vel_field_data)
        plt.clf()
        plot_vel_field(dataset.__getitem__(0)[0], dataset.__getitem__(0)[1], flow_name="aftrain_original", path=cfg.timm_path)
        wandb.log({"aftrain_original_sample": wandb.Image(cfg.timm_path+"aftrain_original.png")})
        test_reconstruction_plot(dataset, bestModel, n_samples=cfg.n_samples, fname="timm_aftrain_test_sample", path=cfg.timm_path)





if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    args = parser.parse_args()

    ARGS_MODE = args.mode

    if ARGS_MODE=='single_run':
        main()
    else:
        sweep_id = wandb.sweep(sweep=hpt_config, project="s'jar")
        wandb.agent(sweep_id, function=main, count=30)


# END        

