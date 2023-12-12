import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from timeit import default_timer as timer

from src_utils import create_envEnc_dtDec_dataset, setup_env
from src_utils import get_data_split, generate_square_subsequent_mask, visualize_input
from src_utils import see_steplr_trend, simulate_tgt_actions, plot_attention_weights, checkpoint_model
from src_utils import convert_angle_to_vectors, plot_grad_flow, get_next_item
from utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object, make_dir
from custom_models import EnvEnc_dtDec_Transformer_v1

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
import tqdm
from utils import execution_time
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from GPT_paper_plots import paper_plots

wandb.login()

DATASET_CREATION_MAP = {
                        # "DOLS": create_action_dataset_v2,
                        # "GenHW": create_action_dataset_v3,
                        # "GPT_dset": verify TODO
                        }

# TODO: 
#       2.  can write another function to return optimzer - see class_optimsAndScheds.py
#       3.  Figure out scale_velocity issue # Done, False when processing and in trasnlating as well
#       5.  Better scheduling behaviour possible? step() at update instead of epoch?
@execution_time
def check_dataloader_time(model, optimizer, train_dataloader, cfg, args, scheduler=None, log_interval=50, transfer=False):
    model.train()
    avg_loss = 0
    loss = 0
    count = 0
    device = cfg.device
    # for env_coef_seq, tgt in train_dataloader:
    # timesteps, actions, states, rtg, action_mask, target_state, encoder_input, n, idx, flow_dir, rzn
    for _, (timesteps, actions, states, rtg, action_mask, final_pos, encoder_input, tgt_padding_mask, n, idx, flow_dir, rzn) in enumerate(tqdm.tqdm(train_dataloader)):
        if transfer:
            timesteps = timesteps.to(device)    # B x T                (,121)
            states = states.to(device)          # B x T x state_dim  (,122,3)
            rtg = rtg.to(device) # B x T x 1                        (,121)
            action_mask = action_mask.to(device)    # B x T           (,121)
            encoder_input = encoder_input.to(device)#                (,120,768)
            final_pos = final_pos.to(device)        #                    (,2)
            act_dim = actions.shape[-1]
            actions = actions.to(device)             #                   (,121)
            action_target = torch.clone(actions).detach().to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)
        pass
    #     action_preds = model.forward(   src=encoder_input,
    #                                     tgt_states=states,
    #                                     tgt_actions=actions,
    #                                     tgt_rtg=rtg,
    #                                     tgt_final_pos=final_pos,
    #                                     src_mask=None,
    #                                     tgt_mask=generate_square_subsequent_mask(model.max_dec_len, device),
    #                                     src_padding_mask=None,
    #                                     tgt_padding_mask=tgt_padding_mask,
    #                                     memory_key_padding_mask=None,
    #                                     timesteps=timesteps,
    #                                     )
        
    #     # action_mask = [F,F,F..... T,T,T], padding is done with T
    #     action_preds = action_preds.view(-1, act_dim)[action_mask.view(-1,) == 0]
    #     action_target = action_target.view(-1, act_dim)[action_mask.view(-1,) == 0]        
        
    #     # loss = F.mse_loss(action_preds, action_target, reduction='mean')
    #     if cfg.loss_type == 'mse_over_angle':
    #         loss = F.mse_loss(action_preds, action_target, reduction='mean')
    #     elif cfg.loss_type == 'mean_norm_of_vec_diff':
    #         logits_vec2d = convert_angle_to_vectors(action_preds, device=cfg.device)
    #         tgt_out_vec2d = convert_angle_to_vectors(action_target, device=cfg.device)
    #         loss = torch.mean(torch.linalg.norm(logits_vec2d - tgt_out_vec2d,axis=1))
        
    #     optimizer.zero_grad()
    #     loss.backward()
    #     plot_grad_flow(model.named_parameters())
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    #     optimizer.step()
    #     scheduler.step()
        
    #     avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))

    #     if count % log_interval == 0:
    #         # param_norms = [p.grad.data.norm(2).item() for p in model.parameters()]
    #         # mean_norm = np.mean(param_norms)
    #         # min_norm = np.min(param_norms)
    #         # max_norm = np.max(param_norms)
    #         wandb.log({f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
    #                     f"lr":  scheduler.get_last_lr()[0],
    #                     # "in_eval/MAX_param_norm": max_norm,
    #                     # "in_eval/MIN_param_norm": min_norm,
    #                     # "in_eval/AVG_param_norm": mean_norm,
    #                    })
    #     count += 1

    # all_att_mats = None
    return 


@execution_time
def train_epoch(model, optimizer, train_dataloader, cfg, args, scheduler=None, log_interval=50):
    model.train()
    device = cfg.device
    avg_loss = torch.tensor(0).to(device)
    loss = None
    count = torch.tensor(0).to(device)
    # for env_coef_seq, tgt in train_dataloader:
    # timesteps, actions, states, rtg, action_mask, target_state, encoder_input, n, idx, flow_dir, rzn
    for k, (timesteps, actions, states, rtg, action_mask, final_pos, encoder_input, tgt_padding_mask, n, idx, flow_dir, rzn) in enumerate(tqdm.tqdm(train_dataloader)):
        timesteps = timesteps.to(device)    # B x T                (,121)
        states = states.to(device)          # B x T x state_dim  (,122,3)
        rtg = rtg.to(device) # B x T x 1                        (,121)
        action_mask = action_mask.to(device)    # B x T           (,121)
        encoder_input = encoder_input.to(device)#                (,120,768)
        final_pos = final_pos.to(device)        #                    (,2)
        act_dim = actions.shape[-1]
        actions = actions.to(device)             #                   (,121)
        action_target = torch.clone(actions).detach().to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)
        action_preds = model.forward(   src=encoder_input,
                                        tgt_states=states,
                                        tgt_actions=actions,
                                        tgt_rtg=rtg,
                                        tgt_final_pos=final_pos,
                                        src_mask=None,
                                        tgt_mask=generate_square_subsequent_mask(model.max_dec_len, device),
                                        src_padding_mask=None,
                                        tgt_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=None,
                                        timesteps=timesteps,
                                        )
        
        # action_mask = [F,F,F..... T,T,T], padding is done with T
        action_preds = action_preds.view(-1, act_dim)[action_mask.view(-1,) == 0]
        action_target = action_target.view(-1, act_dim)[action_mask.view(-1,) == 0]        
        
        # loss = F.mse_loss(action_preds, action_target, reduction='mean')
        if cfg.loss_type == 'mse_over_angle':
            loss = F.mse_loss(action_preds, action_target, reduction='mean')
        elif cfg.loss_type == 'mean_norm_of_vec_diff':
            logits_vec2d = convert_angle_to_vectors(action_preds, device=cfg.device)
            tgt_out_vec2d = convert_angle_to_vectors(action_target, device=cfg.device)
            loss = torch.mean(torch.linalg.norm(logits_vec2d - tgt_out_vec2d,axis=1)/(2*torch.pi))
        
        optimizer.zero_grad()
        loss.backward()
        # plot_grad_flow(model.named_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        # scheduler.step()
        
        avg_loss = avg_loss + ((loss - avg_loss)/(count+1))

        if count % log_interval == 0:
            # param_norms = [p.grad.data.norm(2).item() for p in model.parameters()]
            # mean_norm = np.mean(param_norms)
            # min_norm = np.min(param_norms)
            # max_norm = np.max(param_norms)
            wandb.log({
                        # f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
                        f"lr":  scheduler.get_last_lr()[0],
                        # "in_eval/MAX_param_norm": max_norm,
                        # "in_eval/MIN_param_norm": min_norm,
                        # "in_eval/AVG_param_norm": mean_norm,
                       })
        count += 1

    all_att_mats = None
    return avg_loss, all_att_mats


def evaluate(model, val_dataloader, cfg, log_interval=10):
    model.eval()
    device = cfg.device
    avg_loss = torch.tensor(0).to(device)
    loss = None
    count = torch.tensor(0).to(device)
    with torch.no_grad():
        # for env_coef_seq, tgt in train_dataloader:
        for k, (timesteps, actions, states, rtg, action_mask, final_pos, encoder_input, tgt_padding_mask, n, idx, flow_dir, rzn) in enumerate(tqdm.tqdm(val_dataloader)):#val_dataloader:
            timesteps = timesteps.to(device)    # B x T                (,121)
            states = states.to(device)          # B x T x state_dim  (,122,3)
            rtg = rtg.to(device) # B x T x 1                        (,121)
            action_mask = action_mask.to(device)    # B x T           (,121)
            encoder_input = encoder_input.to(device) #                (,120,768)
            final_pos = final_pos.to(device)         #                    (,2)
            act_dim = actions.shape[-1]
            actions = actions.to(device)             #                   (,121)
            action_target = torch.clone(actions).detach().to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            action_preds = model.forward(   src=encoder_input,
                                            tgt_states=states,
                                            tgt_actions=actions,
                                            tgt_rtg=rtg,
                                            tgt_final_pos=final_pos,
                                            src_mask=None,
                                            tgt_mask=generate_square_subsequent_mask(model.max_dec_len, device),
                                            src_padding_mask=None,
                                            tgt_padding_mask=tgt_padding_mask,
                                            memory_key_padding_mask=None,
                                            timesteps=timesteps,
                                            )
            
            # action_mask = [F,F,F..... T,T,T], padding is done with T
            action_preds = action_preds.view(-1, act_dim)[action_mask.view(-1,) == 0]
            action_target = action_target.view(-1, act_dim)[action_mask.view(-1,) == 0]        
            
            # loss = F.mse_loss(action_preds, action_target, reduction='mean')
            if cfg.loss_type == 'mse_over_angle':
                loss = F.mse_loss(action_preds, action_target, reduction='mean')
            elif cfg.loss_type == 'mean_norm_of_vec_diff':
                logits_vec2d = convert_angle_to_vectors(action_preds, device=cfg.device)
                tgt_out_vec2d = convert_angle_to_vectors(action_target, device=cfg.device)
                loss = torch.mean(torch.linalg.norm(logits_vec2d - tgt_out_vec2d,axis=1)/(2*torch.pi))
            
            # losses += loss.item()
            avg_loss = avg_loss + ((loss - avg_loss)/(count+1))
            # if count%log_interval == 0:
            #     wandb.log({f"in_eval/avg_val_loss vs log_intervalth update": avg_loss})
            count += 1

        # all_att_mats = extract_attention_scores(model)
    return avg_loss, None


def train_model(args=None, cfg_name=None):

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

    wandb_exp_name = "envEnc_dtDec_" + dataset_name + "__" + start_time_str
    wb_id = wandb.util.generate_id()
    config['wb_id'] = wb_id
    wandb.init(project = "envEnc_dtDec",
                name = wandb_exp_name,
                config=config,
                id = wb_id,
                resume = "allow")
    cfg=wandb.config
    # params2 = read_cfg_file(cfg_name=join(ROOT,cfg.params2_name))

    lr = cfg.lr    
    num_epochs = cfg.num_epochs
    context_len = cfg.context_len     # K in decision transformer
    device = torch.device(cfg.device)

    # load data from this file
    dataset_path = join(ROOT,cfg.dataset_path)
    dataset_name = cfg.dataset_name
    # save experiment data in this directory
    log_dir = join(ROOT, cfg.log_dir)
    
    if ARGS_QR:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        num_epochs = 2

    prefix = "my_EnvE_DtD_" + dataset_name
    model_save_dir = join(log_dir, prefix + "_model_" + start_time_str)
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
    print("dataset path: " + dataset_path)
    print("model save path: " + model_save_dir)
    print("\n ------ \n "*2)
    
    # Load and Split dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    if hasattr(cfg, 'mae_model_name'):
        autoenc = torch.load(cfg.mae_model_name)
        ae_type = 'mae'
    elif hasattr(cfg, 'tae_model_arch'):
        # autoenc = torch.load(cfg.tae_model_arch)
        # autoenc.load_state_dict(torch.load(cfg.tae_state_dict))
        from finetune_tinyautoencoder import TAESD, Clamp, Block, conv, Encoder, Decoder
        autoenc = TAESD()
        ae_type = 'tae'
    else:
        raise ValueError("Invalide cfg keys for autoencoder")
    idx_split, set_split = get_data_split(traj_dataset,
                                        split_ratio=cfg.split_tr_tst_val, 
                                        random_seed=cfg.split_ran_seed, 
                                        random_split=cfg.random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split

    # dataset contains optimal actions for different realizations of the env
    tr_set = create_envEnc_dtDec_dataset(train_traj_set, 
                                   train_idx_set,
                                   context_len,
                                   autoenc,
                                   cfg.rtg_scale,
                                   ae_type=ae_type,
                                   device = device)
    states_stats = tr_set.get_states_stats()
    states_stats_path = join(model_save_dir,"states_stats.npy")
    np.save(states_stats_path, states_stats)
    
    val_set = create_envEnc_dtDec_dataset(val_traj_set, 
                            val_idx_set,
                            context_len,
                            autoenc,
                            cfg.rtg_scale,
                            norm_params_4_val = states_stats,
                            ae_type=ae_type,
                            device = device)

    # TODO: can remove this later
    test_set = create_envEnc_dtDec_dataset(test_traj_set, 
                            test_idx_set,
                            context_len, 
                            autoenc,
                            cfg.rtg_scale,
                            norm_params_4_val = states_stats,
                            ae_type=ae_type,
                            device = device)
   
    tr_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size, shuffle=True) #, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True) #, num_workers=cfg.num_workers)
    
    
    #  timesteps, actions, states, rtg, action_mask, target_state, encoder_input, tgt_padding_mask, n, idx, flow_dir, rzn
    _, dummy_actions, dummy_states, dummy_rtg, _, _, dummy_enc_ip, _, _, _,dummy_flow_dir,_ = tr_set[0]

    # intantiate gym env for vizualization purposes
    # if cfg.viz_input:
    #     env_4_viz = setup_env(dummy_flow_dir)
    #     visualize_input(val_set, log_wandb=True, at_time=119, env=env_4_viz, break_at=cfg.viz_break_at)
    #     if cfg.sim_tgt_actions:
    #         simulate_tgt_actions(val_set,
    #                                 env=env_4_viz,
    #                                 log_wandb=True,
    #                                 wandb_fname='simulate_tgt_actions',
    #                                 plot_flow=True,
    #                                 at_time=119,
    #                                 break_at=cfg.viz_break_at)
            
    src_vec_dim = dummy_enc_ip.shape[-1]
    tgt_action_dim = dummy_actions.shape[-1]
    tgt_states_dim = dummy_states.shape[-1]
    transformer = EnvEnc_dtDec_Transformer_v1(
                src_vec_dim=src_vec_dim,
                state_dim=tgt_states_dim,
                action_dim=tgt_action_dim,
                max_tsteps=cfg.context_len,
                num_encoder_layers=cfg.num_encoder_layers,
                num_decoder_layers=cfg.num_decoder_layers,
                emb_size=cfg.embed_dim,
                nhead=cfg.n_heads,
                dropout=cfg.dropout_p,
                positional_encoding=cfg.pos_encoding_type,
                                                ).to(device)
    
    if cfg.optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
                        transformer.parameters(),
                        lr=lr,
                        weight_decay=cfg.wt_decay)
    elif cfg.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(transformer.parameters(), 
                                    lr=lr, 
                                    weight_decay=cfg.wt_decay,
                                    betas=(0.9, 0.98), 
                                    eps=1e-9, )    
    # main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
    #                                                                T_max=num_epochs-3, 
    #                                                                eta_min = 0.0
    # )
    
    # define a scheduler with warmup 
    
    
    
    
    gamma, _ = see_steplr_trend(step_size=cfg.stepLR_stepsize, num_epochs=num_epochs, 
                                lr=lr, final_lr=cfg.final_lr, 
                                update_inside_epoch=False, len_tr_set=len(tr_set),
                                show_plot=True)
    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.stepLR_stepsize, gamma=gamma)
    
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                          start_factor=0.033,
                                                          total_iters=3)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                      schedulers=[warm_up_scheduler, main_lr_scheduler],
                                                      milestones=[3])
    # for _ in range(5):
    #     check_dataloader_time(transformer, optimizer, tr_dataloader, cfg, args, scheduler=scheduler, transfer=True)
    #     check_dataloader_time(transformer, optimizer, tr_dataloader, cfg, args, scheduler=scheduler, transfer=False)

   
    print(f" {src_vec_dim=} \n {tgt_action_dim=} \n {tgt_states_dim=}")
    pytorch_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

    chkpt_list = cfg.chkpt_list if len(cfg.chkpt_list)>0 else [i for i in range(cfg.chkpt_interval, num_epochs, cfg.chkpt_interval)]
    min_vloss = float('inf')
    # save_model_architecture
    checkpoint_model(transformer, model_save_dir, None, optimizer,
            scheduler=scheduler, only_save_states=False)
    
    for epoch in range(0, num_epochs):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        print("\ttraining")
        train_loss, _ = train_epoch(transformer, optimizer, tr_dataloader, cfg, args, scheduler=scheduler)
        epoch_end_time = timer()
        print("\tevaluating")
        val_loss, _ = evaluate(transformer, val_dataloader, cfg)
        print("\tfinished evaluating")
        scheduler.step()
        wandb.log({f"lr":  scheduler.get_last_lr()[0] })
        
        # Evalutation by translation   
        # TODO: can we run non blocking evaluate_pptt.py from here?
        if epoch % cfg.eval_inerval == 0:
            pass

        log_dict = {    "epoch": epoch,   
                        "tr_loss_vs_epoch (unpadded elems)": train_loss,
                        "val_loss_vs_epoch (unpadded elems)": val_loss,
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
                checkpoint_model(transformer, model_save_dir, epoch, optimizer,
                        scheduler=scheduler,
                        loss=(train_loss, val_loss),
                        save_type='some_epoch',)
                
        if epoch == cfg.break_training:
                    break
        

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved last updated model at: " + model_save_dir)
    print("=" * 60)

    return 

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
    parser.add_argument('--CFG', type=str, default='v5_GPT_DG3_pptt')
    args = parser.parse_args()

    cfg_name = "cfg/contGrid_" + args.CFG
    sweep_cfg_name = cfg_name + "_sweep"
    cfg_name =  cfg_name + ".yaml"
    sweep_cfg_name =  sweep_cfg_name + ".yaml"
    cfg_name = join(ROOT,cfg_name)
    sweep_cfg_name = join(ROOT,sweep_cfg_name)

    ARGS_QR = args.quick_run # for sweep mode
    ARGS_CFG = args.CFG
    print(f'args.mode = {args.mode}')
    print(f"ARGS_QR={ARGS_QR}")    

    if args.mode == 'single_run':
        print("----- beginning single_run ------")
        train_model(args, cfg_name)

    elif args.mode == 'sweep':
        print("----- beginning sweeeeeeeeeeeeep ------")
        sweep_cfg = read_cfg_file(cfg_name=sweep_cfg_name)
        sweep_id = wandb.sweep(sweep_cfg)
        wandb.agent(sweep_id, function=train_model)