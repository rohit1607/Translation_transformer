import os


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from timeit import default_timer as timer

from src_utils import create_action_dataset_v5, compare_trajectories, viz_op_traj_with_attention
from src_utils import get_data_split, create_mask, denormalize, visualize_output, visualize_input
from src_utils import see_steplr_trend, simulate_tgt_actions, plot_attention_weights, setup_env
from src_utils import convert_angle_to_vectors
from utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object, show_num_of_params
from utils import make_dir
from custom_models import e2e_Seq2SeqTransformer_v1
from train_mae import MAE, ViT, Transformer, PreNorm, FeedForward, Attention

import gym
import gym_examples
import sys
import pickle
import wandb
import imageio.v2 as imageio
import time
import argparse
from os.path import join
from datetime import datetime
import numpy as np
from root_path import ROOT

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from paper_plots import paper_plots

# for multi-GPU
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group




wandb.login()

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12355"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    

def extract_attention_scores(model):
    enc_sa_arr = np.array([layer.enc_avg_att_scores.cpu().detach().numpy() for layer in model.transformer.encoder.layers])
    dec_sa_arr = np.array([layer.dec_avg_att_scores.cpu().detach().numpy() for layer in model.transformer.decoder.layers])
    dec_ga_arr = np.array([layer.dec_avg_cross_att_scores.cpu().detach().numpy() for layer in model.transformer.decoder.layers])
    return enc_sa_arr, dec_sa_arr, dec_ga_arr 

def plot_all_attention_mats(all_att_mats, log_wandb=True, model_name=''):
    enc_sa_arr , dec_sa_arr, dec_ga_arr = all_att_mats
    save_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tmp/last_exp_figs"

    plot_attention_weights(enc_sa_arr, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=True,
                            scale_each_row=True, 
                            causal_mask_used=False,
                            log_wandb = log_wandb,
                            fname=join(save_path,'attention_heatmap_'+model_name),
                            info_string = 'enc_sa',
                            wandb_fname = 'enc_sa'   
                            )
    plot_attention_weights(dec_sa_arr, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=True,
                            scale_each_row=True, 
                            causal_mask_used=True,
                            log_wandb = log_wandb,
                            fname=join(save_path,'attention_heatmap'+model_name),
                            info_string = 'dec_sa',
                            wandb_fname = 'dec_sa'                 
                            )
    plot_attention_weights(dec_ga_arr, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=True,
                            scale_each_row=True, 
                            causal_mask_used=False,
                            log_wandb = log_wandb,
                            fname=join(save_path,'attention_heatmap'+model_name),
                            info_string = 'dec_ga',
                            wandb_fname = 'dec_ga'                 
                            )
    
    return 

def train_epoch(model, optimizer, train_dataloader, cfg, args, scheduler=None, log_interval=50, gpu_id=None):
    model.train()
    # losses = 0
    avg_loss = 0

    loss = 0
    count=  0
    # for vel_fld_seq, tgt in train_dataloader:
    for timesteps, tgt, traj_mask, loc_tensor, vel_fld_seq, traj_len, idx, _, _ in train_dataloader:
        timesteps = timesteps.to(gpu_id)
        src = vel_fld_seq[:,0:cfg.context_len - 1].to(gpu_id)
        tgt = tgt.to(gpu_id)
        loc_tensor = loc_tensor.to(gpu_id)
        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(gpu_id)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, gpu_id, extra_tokens=1)

        logits = model(src, loc_tensor, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, timesteps)

        tgt_out = tgt[:, 1:, :]
        mask_for_loss = torch.clone(tgt_padding_mask).to(gpu_id)
        mask_for_loss[:,:-1] = ~tgt_padding_mask[:,1:]
        mask_for_loss[:,-1] = tgt_padding_mask[:,0]
        # mask_for_loss = torch.cat([~tgt_padding_mask[:,1:],tgt_padding_mask[:,0]])
       
        # only consider non padded elements (except one at the end)
        logits_ =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        tgt_out_ = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]

        # only considers purely non-padded elements in predictions
        logits =  logits.view(-1,1)[mask_for_loss.view(-1,)]
        tgt_out = tgt_out.reshape(-1,1)[mask_for_loss.view(-1,)]
        # Consider all elements across context length
        # logits =  logits.view(-1,1)
        # tgt_out = tgt_out.reshape(-1,1)
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        # TODO: restrict output to 0-6 or 0-1 if scaled 
        if cfg.loss_type == 'mse_over_angle':
            loss = F.mse_loss(logits, tgt_out)
        elif cfg.loss_type == 'mean_norm_of_vec_diff':
            logits_vec2d = convert_angle_to_vectors(logits, device=gpu_id)
            tgt_out_vec2d = convert_angle_to_vectors(tgt_out, device=gpu_id)
            loss = torch.mean(torch.linalg.norm(logits_vec2d - tgt_out_vec2d,axis=1))
        
        optimizer.zero_grad()

        loss.backward()
        # TODO: try 10
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        # if not scheduler == None:
        #     scheduler.step()
        # losses += loss.item()
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))

        if count%log_interval == 0:
            param_norms = [p.grad.data.norm(2).item() for p in model.parameters() if not p.grad == None]
            mean_norm = np.mean(param_norms)
            min_norm = np.min(param_norms)
            max_norm = np.max(param_norms)
            wandb.log({f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
                        "in_eval/MAX_param_norm": max_norm,
                        "in_eval/MIN_param_norm": min_norm,
                        "in_eval/AVG_param_norm": mean_norm,
                       })
        count += 1

    all_att_mats = extract_attention_scores(model.module)
    return avg_loss, all_att_mats



def evaluate(model, val_dataloader, cfg, log_interval=10, gpu_id=None):
    model.eval()
    # losses = 0
    avg_loss = 0


    count=  0
    for timesteps, tgt, traj_mask, loc_tensor, vel_fld_seq, traj_len, idx, _, _ in val_dataloader:
        timesteps = timesteps.to(gpu_id)
        src = vel_fld_seq[:,0:cfg.context_len - 1].to(gpu_id)
        tgt = tgt.to(gpu_id)
        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(gpu_id)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, gpu_id, extra_tokens=1)
        loc_tensor = loc_tensor.to(gpu_id)

        logits = model(src, loc_tensor, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, timesteps)
        
        tgt_out = tgt[:, 1:, :]
        mask_for_loss = torch.clone(tgt_padding_mask).to(gpu_id)
        mask_for_loss[:,:-1] = ~tgt_padding_mask[:,1:]
        mask_for_loss[:,-1] = tgt_padding_mask[:,0]
        # mask_for_loss = torch.cat([~tgt_padding_mask[:,1:],tgt_padding_mask[:,0]])
       
        # only consider non padded elements (except one at the end)
        logits_ =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        tgt_out_ = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]

        # only considers purely non-padded elements in predictions
        logits =  logits.view(-1,1)[mask_for_loss.view(-1,)]
        tgt_out = tgt_out.reshape(-1,1)[mask_for_loss.view(-1,)]

        # logits =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        # tgt_out = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        if cfg.loss_type == 'mse_over_angle':
            loss = F.mse_loss(logits, tgt_out)
        elif cfg.loss_type == 'mean_norm_of_vec_diff':
            logits_vec2d = convert_angle_to_vectors(logits, device=gpu_id)
            tgt_out_vec2d = convert_angle_to_vectors(tgt_out, device=gpu_id)
            loss = torch.mean(torch.linalg.norm(logits_vec2d - tgt_out_vec2d,axis=1))
        
        # losses += loss.item()
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))
        if count%log_interval == 0:
            wandb.log({f"in_eval/avg_val_loss vs log_intervalth update": avg_loss})
        count += 1

    all_att_mats = extract_attention_scores(model.module)
    return avg_loss, all_att_mats



def train_model(rank: int, world_size: int, args=None, cfg_name=None):
    
    if args.multiGPU:
        ddp_setup(rank, world_size)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M-%S")
    
    if cfg_name != None:
        cfg = read_cfg_file(cfg_name=cfg_name) 
        config = cfg
        dataset_name = cfg['dataset_name']
    else:
        defaults = dict()
        config = defaults
        dataset_name = NAME_MAP[ARGS_CFG] #for sweep mode


    wandb_exp_name = "env2a_" + dataset_name + "__" + start_time_str
    
    wandb.init(project="translation-transformer",
        name = wandb_exp_name,
        config=config,
        )

    cfg=wandb.config
    # cfg_copy = cfg


    add_trans_noise = cfg.add_transition_noise_during_inf

    env_name = cfg.env_name
    split_tr_tst_val = cfg.split_tr_tst_val
    split_ran_seed = cfg.split_ran_seed
    random_split = cfg.random_split
    max_eval_ep_len = cfg.max_eval_ep_len  # max len of one episode
    num_eval_ep = cfg.num_eval_ep       # num of evaluation episodes


    batch_size = cfg.batch_size           # training batch size
    optimizer_name = cfg.optimizer_name
    lr = cfg.lr    
    final_lr = cfg.final_lr                        # const learning rate
    wt_decay = cfg.wt_decay               # weight decay
    use_scheduler = cfg.use_scheduler
    warmup_steps = cfg.warmup_steps       # warmup steps for lr scheduler

    # total_updates = max_train_iters x num_updates_per_iter
    num_epochs = cfg.num_epochs
    # num_updates_per_iter = cfg.num_updates_per_iter
    # comp_val_loss = cfg.comp_val_loss
    eval_inerval =  cfg.eval_inerval
    target_conditioning = cfg.target_conditioning
    num_encoder_layers = cfg.num_encoder_layers
    num_decoder_layers = cfg.num_decoder_layers
    context_len = cfg.context_len     # K in decision transformer
    req_context_len = cfg.req_context_len 
    embed_dim = cfg.embed_dim          # embedding (hidden) dim of transformer
    n_heads = cfg.n_heads            # num of transformer heads
    dropout_p = cfg.dropout_p         # dropout probability

    # load data from this file
    dataset_path = join(ROOT,cfg.dataset_path)
    dataset_name = cfg.dataset_name
    # saves model and csv in this directory
    log_dir = join(ROOT, cfg.log_dir)
    tt_eb = cfg.translate_earlybreaks
    # TODO: SHUBHAM: make a new entry called mae_name in cfg_v5_GenHW.yaml # DONE
    mae_model_name = cfg.mae_model_name
    # training and evaluation device
    device = torch.device(cfg.device)
    # torch.cuda.empty_cache()

    pp_cfg = {}
    if cfg.preprocessor == 'ViT_scratch':
        pp_cfg['hidden_size'] = cfg.embed_dim #hidden size of vit should be same as emb_dim
        pp_cfg['num_hidden_layers'] = cfg.num_hidden_layers_vit
        pp_cfg['num_attention_heads'] = cfg.num_attention_heads_vit
        pp_cfg['intermediate_size'] = cfg.intermediate_size_vit
        pp_cfg['image_size'] = cfg.image_size_vit
        pp_cfg['patch_size'] = cfg.patch_size_vit
        pp_cfg['num_channels'] = cfg.num_channels_vit
        pp_cfg['hidden_dropout_prob'] = cfg.hidden_dropout_prob_vit
        pp_cfg['attention_probs_dropout_prob'] = cfg.attention_probs_dropout_prob_vit

        
    if ARGS_QR:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        num_epochs = 2

    prefix = "my_translat_" + dataset_name
    save_model_dir = prefix + "_model_" + start_time_str
    save_model_path = join(log_dir, save_model_dir)

    if rank == 0:
        print("=" * 60)
        print("start time: " + start_time_str)
        print("=" * 60)
        print("device set to: " + str(device))
        print("dataset path: " + dataset_path)
        print("model save path: " + save_model_path)

    # env = gym.make(env_name)
    # env.setup(cfg, params2, add_trans_noise=add_trans_noise)
    
    
    # Load and Split dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    # traj_dataset =  traj_dataset[0:500]
    traj_dataset_new = []
    for i in range(len(traj_dataset)):
        a, b = traj_dataset[i][2].shape
        if  a < req_context_len:
            traj_dataset_new.append(traj_dataset[i])
    
    traj_dataset_new =  traj_dataset_new[0:200]    
    # context_len = np.max([(traj_dataset_new[i][2]).shape for i in range(len(traj_dataset_new))]) + 2 
    # wandb.config.update(cfg, allow_val_change=True)
    # cfg['context_len'] = context_len  

    
    idx_split, set_split = get_data_split(traj_dataset_new,
                                        split_ratio=split_tr_tst_val, 
                                        random_seed=split_ran_seed, 
                                        random_split=random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split


    # dataset contains optimal actions for different realizations of the env
    tr_set = create_action_dataset_v5(dataset=train_traj_set, 
                            #train_idx_set,
                            idx_set=train_idx_set,
                            context_len=context_len,
                                        )
    
    # src_stats = tr_set.get_src_stats()
    # src_stats_path = save_model_path[:-3] +"_src_stats.npy"
    # np.save(src_stats_path, src_stats)
    val_set = create_action_dataset_v5(val_traj_set, 
                            val_idx_set,
                            context_len,
                            
                                        )
    test_set = create_action_dataset_v5(test_traj_set, 
                            val_idx_set,
                            context_len,
                            
                                        )

    train_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size, shuffle=False,
                                    sampler=DistributedSampler(tr_set)
                                )
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                                    sampler=DistributedSampler(val_set)
                                )
    # visualize_input(val_set, stats=None, log_wandb=True, at_time=119, info_str='val', color_by_time=False)
    # visualize_input(test_set, stats=None, log_wandb=True, at_time=119, info_str='test', color_by_time=False)

    _, dummy_target, _, _, dummy_vel_fld_seq, _,_,dummy_flow_dir,_ = tr_set[0]
    tgt_vec_dim = dummy_target.shape[-1]
    # intantiate gym env for vizualization purposes
    env_4_viz = setup_env(dummy_flow_dir)

    # visualize_input(tr_set, log_wandb=True, at_time=119, env=env_4_viz, traj_idx=[k for k in range(200)])
    # simulate_tgt_actions(tr_set,
    #                         env=env_4_viz,
    #                         gathered_envs=True,
    #                         log_wandb=True,
    #                         wandb_fname='simulate_tgt_actions',
    #                         plot_flow=True,
    #                         at_time=119,
    #                         plot_range=100)

    
    transformer = e2e_Seq2SeqTransformer_v1(num_encoder_layers, num_decoder_layers, embed_dim,
                                 n_heads, 
                                 embed_dim, #src_vec_dim=embed_dim
                                 tgt_vec_dim, 
                                 preprocessor=cfg.preprocessor,
                                 preprocessor_cfg=pp_cfg,
                                #  device=cfg.device,
                                 device=rank,
                                 dim_feedforward=None,     # TODO: add dim_ffn to cfg
                                 max_len=context_len,
                                 positional_encoding=cfg.pos_encoding_type,
                                 pixel_scale=10,
                                #  ).to(cfg.device)
                                ).to(rank)

    
    # if args.multiGPU:
    transformer = DDP(transformer, device_ids=[rank],find_unused_parameters=True)
 
    
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
                        transformer.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(transformer.parameters(), 
                                    lr=lr, 
                                    weight_decay=wt_decay,
                                    betas=(0.9, 0.98), 
                                    eps=1e-9, 
                                    )   
         
    # main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
    #                                                                T_max=num_epochs-3, 
    #                                                                eta_min = 0.0
    #                                                                 )
    gamma, _ = see_steplr_trend(step_size=20, num_epochs=num_epochs, lr=lr, final_lr=final_lr)
    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                          start_factor=0.033,
                                                          total_iters=3
                                                          )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                      schedulers=[warm_up_scheduler, main_lr_scheduler],
                                                      milestones=[3])


    min_val_loss = float('inf')
    if rank == 0:
        make_dir(save_model_path)
        torch.save(transformer.module, join(save_model_path,'model.pt'))
        cfg_copy_path = join(save_model_path, "used_cfg.yml")
        save_yaml(cfg_copy_path,cfg)
        
    for epoch in range(0, num_epochs+1):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        print("training")
        train_loss, tr_all_att_mat = train_epoch(transformer, optimizer, train_dataloader, cfg, args, scheduler=scheduler,gpu_id=rank)
        epoch_end_time = timer()
        print("evaluating")
        val_loss, val_all_att_mat = evaluate(transformer, val_dataloader, cfg, gpu_id=rank)
        scheduler.step()

        if rank == 0:
            wandb.log({f"in_eval/lr":  scheduler.get_last_lr()[0]
                })
            if val_loss < min_val_loss:
                print("\n --- checkpointing best_model ---\n")
                save_path = join(save_model_path, 'best_model_cp')
                min_val_loss = val_loss
                checkpoint(transformer.module,optimizer,epoch,scheduler,train_loss,val_loss,save_path)
            if epoch in range(0, num_epochs, cfg.model_save_interval):
                print(f"\n --- cehckpoinitng model at {epoch}/{num_epochs} --- \n")
                save_path = join(save_model_path, 'int_saved_model_cp')
                checkpoint(transformer.module,optimizer,epoch,scheduler,train_loss,val_loss,save_path)
                
    destroy_process_group()
        
    return 

def checkpoint(model,optimizer,epoch,scheduler=None,train_loss=None,val_loss=None,save_path='model'):
    save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
    if not scheduler == None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    if not train_loss == None:
        save_dict['train_loss']= train_loss
    if not val_loss == None:
        save_dict['val_loss']= val_loss,
    torch.save(save_dict, save_path)

    

ARGS_MULTIGPU = False
ARGS_QR = False
ARGS_CFG = "_--_"
NAME_MAP = {
            "v5": "DG3",
            "v5_HW": "HW",
            "v5_GPT_DG3": "GPT_DG3",
            "v5_DOLS": "DOLS"
            }

if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    parser.add_argument('--CFG', type=str, default='v5_GenHW')
    parser.add_argument('--multiGPU', type=bool, default=False)
    args = parser.parse_args()

    cfg_name = "cfg/contGrid_" + args.CFG
    sweep_cfg_name = cfg_name + "_sweep"
    cfg_name =  cfg_name + ".yaml"
    sweep_cfg_name =  sweep_cfg_name + ".yaml"
    cfg_name = join(ROOT,cfg_name)
    sweep_cfg_name = join(ROOT,sweep_cfg_name)

    ARGS_QR = args.quick_run # for sweep mode
    ARGS_CFG = args.CFG
    ARGS_MULTIGPU = args.multiGPU
    print(f'args.mode = {args.mode}')
    print(f"ARGS_QR={ARGS_QR}")

    if args.mode == 'load_prev_and_test':
        print("----- beginning load_prev_and_test ------")
        load_prev_and_test(args, cfg_name)        

    if args.mode == 'single_run' and args.multiGPU == False:
        print("----- beginning single_run on single GPU ------")
        train_model(args, cfg_name)
        
    if args.mode == 'single_run' and args.multiGPU == True:
        print("----- beginning single_run on multi GPU using DDP------")
        # train_model(args, cfg_name)
        world_size = torch.cuda.device_count()
        mp.spawn(train_model, args=(world_size, args, cfg_name),
                 nprocs=world_size)
        
    elif args.mode == 'sweep':
        print("----- beginning sweeeeeeeeeeeeep ------")
        sweep_cfg = read_cfg_file(cfg_name=sweep_cfg_name)
        sweep_id = wandb.sweep(sweep_cfg)
        wandb.agent(sweep_id, function=train_model)

