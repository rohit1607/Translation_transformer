import torch
# from model_min_dt import Transformer_causal_encoder
from custom_models import Transformer_causal_encoder
import pickle
from src_utils import create_env_emb_dataset, get_data_split, see_steplr_trend
from src_utils import checkpoint_model
from mae_all_data_load import plot_vel_field_decoder, GiveMe_loaders, load_vel, VelocityDataset
from utils import read_cfg_file, load_pkl, print_dict, save_object, make_dir, save_yaml, convert_dict_to_obj
from Class_optimsAndScheds import Optims_Scheds
from torch.utils.data import DataLoader
import numpy as np
import wandb
from torch.nn import functional as F
from datetime import datetime
import argparse
from os.path import join
from root_path import ROOT
from timeit import default_timer as timer
from finetune_tinyautoencoder import TAESD, Block, Clamp, conv, Encoder, Decoder
import matplotlib.pyplot as plt
import random


"""
TODOs;
1. See if scheduler can be applied at update level
2. use cosw scheduler and see results
3.
"""

wandb.login()

def train_epoch(model, optimizer, train_dataloader, cfg, scheduler=None, log_interval=50):
    model.train()
    avg_loss = 0
    avg_rse_loss = 0
    loss = 0
    count = 0
    # for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, _, _ in train_dataloader:
    for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in train_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_reps[:, :-1, :].to(cfg.device)
        tgt = env_reps[:, 1:, :].to(cfg.device)
        traj_mask = traj_mask.to(cfg.device)
        
        logits = model(timesteps, src, padding_mask=traj_mask)
        loss = F.mse_loss(logits, tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
        optimizer.step()
        # TODO: See if scheduler can be applied at update level
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))
        # https://www.gepsoft.com/gxpt4kb/Chapter10/Section1/SS06.htm
        rse_loss = loss / F.mse_loss(tgt, tgt - torch.mean(tgt, axis=0))
        avg_rse_loss = avg_rse_loss + ((rse_loss.item() - avg_rse_loss)/(count+1))

        # if count%log_interval == 0:
        #     pass
            # TODO: debug (Shubham)
            # param_norms = [p.grad.data.norm(2).item() for p in model.parameters()]
            # mean_norm = np.mean(param_norms)
            # min_norm = np.min(param_norms)
            # max_norm = np.max(param_norms)
            # wandb.log({f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
            #             "in_eval/MAX_param_norm": max_norm,S
            #             "in_eval/MIN_param_norm": min_norm,
            #             "in_eval/AVG_param_norm": mean_norm,
            #            })
        count += 1
        # scheduler.step()S
        # log_dict = { 
        # "lr" : scheduler.get_last_lr()[0] if cfg.use_scheduler else cfg.lr
        # }
        # wandb.log(log_dict)
    # all_att_mats = extract_attention_scores(model)
    all_att_mats = None
    return avg_loss, all_att_mats, avg_rse_loss

def evaluate(model, val_dataloader, cfg, log_interval=50):
    model.eval()
    avg_loss = 0
    avg_rse_loss = 0
    loss = 0
    count = 0

    # for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, _, _ in train_dataloader:
    for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in val_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_reps[:, :-1, :].to(cfg.device)
        tgt = env_reps[:, 1:, :].to(cfg.device)
        traj_mask = traj_mask.to(cfg.device)

        logits = model(timesteps, src, padding_mask=traj_mask)
        loss = F.mse_loss(logits, tgt)
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))
        rse_loss = loss / F.mse_loss(tgt, tgt - torch.mean(tgt, axis=0))
        avg_rse_loss = avg_rse_loss + ((rse_loss.item() - avg_rse_loss)/(count+1))
        if count%log_interval == 0:
            wandb.log({f"in_eval/avg_val_loss vs log_intervalth update": avg_loss})
        count += 1

    # all_att_mats = extract_attention_scores(model)
    all_att_mats = None
    return avg_loss, all_att_mats, avg_rse_loss


def translate(model: torch.nn.Module, test_idx, test_set, tr_set_stats, cfg, earlybreak=10**8):
    model.eval()
    count = 0           # keeps count of total episodes
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    set_output = []
    full_time_mse_list = []

    with torch.no_grad():
        # for sample in range(len(test_set)):
        for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in test_dataloader:
            if idx%100==0:
                print("in translate, idx=", idx)
            sample_results = {}

            timesteps = timesteps.to(cfg.device)
            count += 1
            if count == earlybreak:
                break
            src = env_reps[:, :-1, :].to(cfg.device)
            tgt = env_reps[:, 1:, :].to(cfg.device)
            preds = torch.zeros((1, cfg.context_len, tgt.shape[2]),dtype=torch.float32, device=cfg.device)
            dyn_inputs = torch.zeros((1, cfg.context_len, tgt.shape[2]),dtype=torch.float32, device=cfg.device)
            padding_mask = torch.ones((1,cfg.context_len)).type(torch.bool).to(cfg.device)
 
            loss_post_t = []
            for i in range(1, cfg.context_len-1):
                dyn_inputs[0,0:i,:] = src[0,0:i,:]
                padding_mask[0,:i] = False
                preds = model(timesteps, dyn_inputs, padding_mask=padding_mask)
                loss_post_t.append(F.mse_loss(preds[0,i:,:],tgt[0,i:,:]))

            full_time_mse = F.mse_loss(preds[0,:],tgt[0,:])
            full_time_mse_list.append(full_time_mse)
            sample_results['loss_post_t'] = loss_post_t
            sample_results['full_time_mse'] = full_time_mse
            set_output.append(sample_results)

    avg_results = {}
    avg_results['avg_full_time_mse'] = torch.mean(torch.stack(full_time_mse_list))

    return set_output, avg_results

def plot_translate(model: torch.nn.Module, test_idx, test_set, tr_set_stats, cfg, earlybreak=10**8):
    model.eval()
    count = 0           # keeps count of total episodes
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    set_output = []
    full_time_mse_list = []
    i_list = [90, 100, 110]
    preds_list = [] # contains tuples of format (flow_dir, rzn, i,preds_post_i(obs_till_i),tgts_post_i )
    with torch.no_grad():
        # for sample in range(len(test_set)):
        for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in test_dataloader:
            if idx%100==0:
                print("in translate, idx=", idx)
            sample_results = {}

            timesteps = timesteps.to(cfg.device)
            count += 1
            if count == earlybreak:
                break
            src = env_reps[:, :-1, :].to(cfg.device)
            tgt = env_reps[:, 1:, :].to(cfg.device)
            preds = torch.zeros((1, cfg.context_len, tgt.shape[2]),dtype=torch.float32, device=cfg.device)
            dyn_inputs = torch.zeros((1, cfg.context_len, tgt.shape[2]),dtype=torch.float32, device=cfg.device)
            padding_mask = torch.ones((1,cfg.context_len)).type(torch.bool).to(cfg.device)
 
            loss_post_t = []
            for i in range(1, cfg.context_len-1):
                dyn_inputs[0,0:i,:] = src[0,0:i,:] # dyn_inputs start with env data observed till i
                padding_mask[0,:i] = False
                for p in range(i, cfg.context_len-1):
                    padding_mask[0,p-1] = False
                    ag_pred_at_p = model(timesteps, dyn_inputs, padding_mask=padding_mask)[0,p-1]
                    dyn_inputs[0,p,:] = ag_pred_at_p
                # IMP_NOTE: dyn_inputs contains observed embs till timestep i (exclusive)
                #                      contains autoregressive predictions from i (inclusive) to context_len-1    
                loss_post_t.append(F.mse_loss(dyn_inputs[0,i:,:],tgt[0,i:,:]))
                if i in i_list:
                    preds_list.append((flow_dir, rzn, i, dyn_inputs[0,i:,:], tgt[0,i:,:]))
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
        flow_dir, rzn, i, preds, targets = preds_list[j]
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
            resulting_image_path = log_exp_dir+"/"+f"figs/i_{i}/env_emb_sample_{r}"+".png"
            plt.savefig(resulting_image_path)
            
            


def load_and_translate(args): 
    
    cfg_path = join(args.log_exp_dir, "run_cfg.yml")
    cfg = read_cfg_file(cfg_path)
    cfg = convert_dict_to_obj(cfg)
    
    # Load model architecture
    model_path = join(args.log_exp_dir, "arch.pt")
    transformer = torch.load(model_path)
    
    # Load model weights
    model_state_dict_path = join(args.log_exp_dir, args.model_state)
    transformer.load_state_dict(torch.load(model_state_dict_path)['model_state_dict'])
    
    # Load and Split dataset
    emb_dataset = load_pkl(cfg.dataset_path)[0:62499]
    idx_split, set_split = get_data_split(emb_dataset,
                                        split_ratio=cfg.split_tr_tst_val, 
                                        random_seed=cfg.split_ran_seed, 
                                        random_split=cfg.random_split)
    train_emb_set, test_emb_set, val_emb_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split
    test_set = create_env_emb_dataset(test_emb_set, test_idx_set, cfg.context_len)
    preds_list = plot_translate(transformer, test_idx_set, test_set, None, 
                                                    cfg, earlybreak=cfg.translate_earlybreaks[0])
    decode_plot(preds_list, args.log_exp_dir)
    
    
    return None
    

    
def train_model(args=None, cfg_name=None):
    
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")
    config = read_cfg_file(cfg_name=cfg_name)

    dataset_name = config['dataset_name']
    wandb_exp_name = "emb_env_td_" + dataset_name + "__" + start_time_str
    wandb.init(project="td_emb",
                name = wandb_exp_name,
                config = config    
                )

    cfg=wandb.config
    lr = cfg.lr
    tt_eb = cfg.translate_earlybreaks 
    device = torch.device(cfg.device)

    prefix = "EmbFcast_" + dataset_name
    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    model_save_dir = join(cfg.log_dir, save_model_name)
    make_dir(model_save_dir)
    
    # save cfg to use for loading later with same run configuration
    wb_cfg_copy_path = join(model_save_dir, "wb_run_cfg.yml")
    save_yaml(wb_cfg_copy_path, cfg)
    cfg_copy_path = join(model_save_dir, "run_cfg.yml")
    save_yaml(cfg_copy_path, config)
    
    print(f"{cfg_copy_path=}\n")
    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + cfg.dataset_path)
    print("model save path: " + model_save_dir)
    
    # Load and Split dataset
    emb_dataset = load_pkl(cfg.dataset_path)[0:62499]
    idx_split, set_split = get_data_split(emb_dataset,
                                        split_ratio=cfg.split_tr_tst_val, 
                                        random_seed=cfg.split_ran_seed, 
                                        random_split=cfg.random_split)
    train_emb_set, test_emb_set, val_emb_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split
    tr_set = create_env_emb_dataset(train_emb_set, train_idx_set, cfg.context_len)
    val_set = create_env_emb_dataset(val_emb_set, val_idx_set, cfg.context_len)
    test_set = create_env_emb_dataset(test_emb_set, test_idx_set, cfg.context_len)
    tr_dataloader = DataLoader(tr_set, batch_size=cfg.bs, shuffle=True,
                                    #   num_workers=10, pin_memory=True
                                    )
    val_dataloader = DataLoader(val_set, batch_size=cfg.bs, shuffle=True,
                                    #   num_workers=10, pin_memory=True
                                    )
    test_dataloader = DataLoader(test_set, batch_size=cfg.bs, shuffle=True,
                                    #   num_workers=10, pin_memory=True
                                    )

    timesteps, env_reps, traj_mask, idx, flow_dir, rzn = tr_set[0]
    src_vec_dim = env_reps.shape[-1] 
    print(f"src_vec_dim = {src_vec_dim}")

    transformer = Transformer_causal_encoder(src_vec_dim, cfg.n_blocks,
                                            cfg.h_dim, cfg.context_len, cfg.drop_p, cfg.pos_encoding_type, cfg.n_heads, cfg.drop_p, 
                                            device=cfg.device).to(cfg.device)
    
    
    obj_OS = Optims_Scheds(model=transformer, optim=cfg.optimizer, sched=cfg.scheduler, cfg=cfg)
    obj_OS.assign()
    optimizer = obj_OS.optimizer()
    scheduler = obj_OS.scheduler()
    
    ## ------------ Commented by Shubham -------------
    # optimizer = torch.optim.AdamW(
    #                 transformer.parameters(),
    #                 lr=lr,
    #                 weight_decay=cfg.wt_decay
    #             ) 
    
    # if optimizer_name == 'AdamW':
    #     optimizer = torch.optim.AdamW(
    #                     transformer.parameters(),
    #                     lr=lr,
    #                     weight_decay=cfg.wt_decay
    #                 )
    # elif optimizer_name == 'Adam':
    #     optimizer = torch.optim.Adam(transformer.parameters(), 
    #                                 lr=lr, 
    #                                 weight_decay=cfg.wt_decay,
    #                                 betas=(0.9, 0.98), 
    #                                 eps=1e-9, 
    #                                 )    
    # main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
    #                                                                T_max=num_epochs-3, 
    #                                                                eta_min = 0.0
    #                                                                 )
    # gamma, _ = see_steplr_trend(step_size=20, num_epochs=cfg.num_epochs, lr=lr, final_lr=cfg.final_lr)
    # main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    
    # warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
    #                                                       start_factor=0.033,
    #                                                       total_iters=3
    #                                                       )
    
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
    #                                                   schedulers=[warm_up_scheduler, main_lr_scheduler],
    #                                                   milestones=[3])
    # ----------------------------------------------

    pytorch_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

 
    max_avg = 10000
    min_vloss = 10000
    chkpt_list = cfg.chkpt_list if len(cfg.chkpt_list)>0 else [i for i in range(cfg.chkpt_interval, cfg.num_epochs, cfg.chkpt_interval)]
    # save_model_architecture
    checkpoint_model(transformer, model_save_dir, None, optimizer,
            scheduler=scheduler, only_save_states=False)
    for epoch in range(0, cfg.num_epochs+1):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        print("training")
        train_loss, _, train_rse_loss = train_epoch(transformer, optimizer, tr_dataloader, cfg, scheduler=scheduler)
        epoch_end_time = timer()
        print("evaluating")
        val_loss, _ , val_rse_loss= evaluate(transformer, val_dataloader, cfg)
        scheduler.step()

        log_dict = { 
            "tr_loss": train_loss,
            "val_loss": val_loss,
            "tr_rse_loss": train_rse_loss,
            "val_rse_loss": val_rse_loss,
            "lr" : scheduler.get_last_lr()[0] if cfg.use_scheduler else lr
            }
        wandb.log(log_dict)

        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
        print('='*60)
        print(f"Epoch: {epoch}")
        for key, val in log_dict.items():
            print(f"{key}: {format(val,'.4f')}")
        print(f"Epoch runtime = {(epoch_end_time - epoch_start_time):.3f}s")
        print(f"time elapsed: {time_elapsed}")
        print("")
        
        # model checkpointing
        if val_loss < min_vloss:
            min_vloss = val_loss
            print("saving current model at: " + model_save_dir)
            checkpoint_model(transformer, model_save_dir, epoch, optimizer,
                             scheduler=scheduler,
                             loss=(train_loss, val_loss),
                             save_type='cur_best',)
        else:
            if epoch in chkpt_list:
                # Evaluation by translation   
                print("translating")
                _, tr_avg_results = translate(transformer, train_idx_set, tr_set, None, 
                                                    cfg, earlybreak=tt_eb[0])
                _, val_avg_results = translate(transformer, val_idx_set, tr_set, None, 
                                                            cfg, earlybreak=tt_eb[1])

                tr_avg_full_time_mse = tr_avg_results['avg_full_time_mse']
                val_avg_full_time_mse = val_avg_results['avg_full_time_mse']
                transl_log_dict = {
                            "translate/epoch": epoch,
                            "translate/tr_avg_full_time_mse": tr_avg_full_time_mse,
                            "translate/val_loss_vs_epoch (unpadded elems)": val_avg_full_time_mse,}
                wandb.log(transl_log_dict)      
                         
                if val_avg_full_time_mse < max_avg:
                    max_avg = val_avg_full_time_mse
                    print("saving current model at: " + model_save_dir)
                    checkpoint_model(transformer, model_save_dir, epoch, optimizer,
                                    scheduler=scheduler,
                                    loss=(train_loss, val_loss),
                                    save_type='cur_best_translation',)   
                else:     
                    checkpoint_model(transformer, model_save_dir, epoch, optimizer,
                    scheduler=scheduler,
                    loss=(train_loss, val_loss),
                    save_type='some_epoch',)    
                        
        
        if epoch == cfg.break_training:
            break
        
        
    # end_time = datetime.now().replace(microsecond=0)
    # end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    # print("started training at: " + start_time_str)
    # print("finished training at: " + end_time_str)
    # print("total training time: " + time_elapsed)
    # print("saved last updated model at: " + save_model_path)
    # print("=" * 60)

    return 



if __name__ == "__main__":
    def_log_exp_dir = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/log/EmbFcast_DG3_model_11-24-11-07.pt"
    
    print(f"cuda available: {torch.cuda.is_available()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_exp_dir', type=str, default=def_log_exp_dir)
    parser.add_argument('--model_state', type=str, default="best_vloss.pt")
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    # parser.add_argument('--load_and_translate', type=bool, default=False)
    parser.add_argument('--CFG', type=str, default='envEmb_predictor')
    args = parser.parse_args()

    cfg_name = "cfg/" + args.CFG
    sweep_cfg_name = cfg_name + "_sweep"
    cfg_name =  cfg_name + ".yaml"
    sweep_cfg_name =  sweep_cfg_name + ".yaml"
    cfg_name = join(ROOT,cfg_name)
    sweep_cfg_name = join(ROOT,sweep_cfg_name)

    ARGS_QR = args.quick_run # for sweep mode
    ARGS_CFG = args.CFG
    print(f'args.mode = {args.mode}')
    print(f"ARGS_QR={ARGS_QR}")

    # if args.mode == 'load_prev_and_test':
    #     print("----- beginning load_prev_and_test ------")
    #     load_prev_and_test(args, cfg_name)        

    if args.mode == 'single_run':
        print("----- beginning single_run ------")
        train_model(args, cfg_name)

    elif args.mode == 'sweep':
        print("----- beginning sweeeeeeeeeeeeep ------")
        sweep_cfg = read_cfg_file(cfg_name=sweep_cfg_name)
        sweep_id = wandb.sweep(sweep_cfg)
        wandb.agent(sweep_id, function=train_model)
        
    elif args.mode =='load_and_translate':
        print('---- loading best model, translating and plotting ----')
        load_and_translate(args)