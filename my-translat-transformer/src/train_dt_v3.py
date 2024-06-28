"""
Apr 10 , 2024
Copied from train_dt
For seeing if dt can fit on perling dataset
- to compare performance with pptt

"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from timeit import default_timer as timer

from src_utils import setup_env, create_dt_dataset_v3, compare_trajectories, viz_op_traj_with_attention
from src_utils import get_data_split, create_mask, denormalize, visualize_output, visualize_input, visualize_actions, visualize_actions_flow
from src_utils import see_steplr_trend, simulate_tgt_actions, plot_attention_weights, convert_angle_to_vectors
from utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object, make_dir, convert_dict_to_obj
from model_min_dt import DecisionTransformer
from paper_plots import paper_plots
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
import os
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from GPT_paper_plots import paper_plots

wandb.login()

def mse_over_angle(pred, tgt):
    return F.mse_loss(pred, tgt, reduction='mean')

def mae_over_angle(pred, tgt):
    return F.l1_loss(pred, tgt, reduction='mean')

def cross_entropy(pred, tgt):
    # pred_logits = torch.logit(pred)
    loss = torch.nn.BCELoss()
    return loss(pred, tgt)

def cos_diff(pred, tgt):
    return 1 - torch.mean(torch.cos((pred - tgt)*2*torch.pi))

def mean_norm_of_vec_diff(pred, tgt, cfg):
    logits_vec2d = convert_angle_to_vectors(pred, device=cfg.device)
    tgt_out_vec2d = convert_angle_to_vectors(tgt, device=cfg.device)
    loss = torch.mean(torch.linalg.norm(logits_vec2d - tgt_out_vec2d,axis=1)/(2*torch.pi))
    return loss

def cosine_similarity(pred, tgt, cfg):
    logits_vec2d = convert_angle_to_vectors(pred, device=cfg.device)
    tgt_out_vec2d = convert_angle_to_vectors(tgt, device=cfg.device)
    return 1- torch.mean(-F.cosine_similarity(logits_vec2d, tgt_out_vec2d, dim=-1))

DATASET_CREATION_MAP = {
                        # "DOLS": create_action_dataset_v2,
                        # "GenHW": create_action_dataset_v3,
                        # "GPT_dset": verify TODO
                        }


def train_epoch(model, optimizer, tr_set, cfg, args, scheduler=None, log_interval=50):
    model.train()
    # losses = 0
    avg_loss = 0
    # TODO: take it outside
    train_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size, shuffle=False)
    loss = 0
    count = 0
    device = cfg.device
    # for env_coef_seq, tgt in train_dataloader:
    for timesteps, actions, traj_mask, target_state, states, traj_len, idx, _, _, returns_to_go in train_dataloader:
        timesteps = timesteps.to(device)    # B x T
        states = states.to(device)          # B x T x state_dim
        returns_to_go = returns_to_go.to(device) # B x T x 1
        traj_mask = traj_mask.to(device)    # B x T
        act_dim = actions.shape[-1]
        actions = actions.to(device)
        action_target = torch.clone(actions).detach().to(device)

        # logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, timesteps)
        state_preds, action_preds, return_preds = model.forward(timesteps=timesteps,
                                                                states=states,
                                                                actions=actions,
                                                                returns_to_go=returns_to_go,
                                                                )
        
        action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
        action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]        
        
        # loss = F.mse_loss(action_preds, action_target, reduction='mean')
        # TODO: write a common function that returns the loss function and use it in both train and eval
        if cfg.loss_type == 'mse_over_angle':
            loss = mse_over_angle(action_preds, action_target)
        elif cfg.loss_type == 'mean_norm_of_vec_diff':
            loss = mean_norm_of_vec_diff(action_preds, action_target, cfg)
        elif cfg.loss_type == 'cosine_similarity':
            loss = cosine_similarity(action_preds, action_target, cfg)
        elif cfg.loss_type == 'cos_diff':
            loss = cos_diff(action_preds, action_target)
        elif cfg.loss_type == 'mae_over_angle':
            loss = mae_over_angle(action_preds, action_target)
        elif cfg.loss_type == "cross_entropy":
            loss = cross_entropy(action_preds, action_target)
        else:
            raise NotImplementedError
        
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
        optimizer.step()
        
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))

        if count % log_interval == 0:
            # param_norms = [p.grad.data.norm(2).item() for p in model.parameters()]
            # mean_norm = np.mean(param_norms)
            # min_norm = np.min(param_norms)
            # max_norm = np.max(param_norms)
            wandb.log({f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
                        # "in_eval/MAX_param_norm": max_norm,
                        # "in_eval/MIN_param_norm": min_norm,
                        # "in_eval/AVG_param_norm": mean_norm,
                       })
        count += 1

    all_att_mats = None
    return avg_loss, all_att_mats


def evaluate(model, val_set, cfg, log_interval=10):
    model.eval()
    # losses = 0
    avg_loss = 0
    bs = cfg.batch_size
    val_dataloader = DataLoader(val_set, batch_size=bs, shuffle=True)

    count = 0
    device = cfg.device
    # for env_coef_seq, tgt in train_dataloader:
    for timesteps, actions, traj_mask, target_state, states, traj_len, idx, _, _, returns_to_go, in val_dataloader:
        timesteps = timesteps.to(device)    # B x T
        states = states.to(device)          # B x T x state_dim
        returns_to_go = returns_to_go.to(device) # B x T x 1
        traj_mask = traj_mask.to(device)    # B x T
        act_dim = actions.shape[-1]
        actions = actions.to(device)
        action_target = torch.clone(actions).detach().to(device)

        # logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, timesteps)
        state_preds, action_preds, return_preds = model.forward(timesteps=timesteps,
                                                                states=states,
                                                                actions=actions,
                                                                returns_to_go=returns_to_go,
                                                                )
        
        action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
        action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]        
        
        # loss = F.mse_loss(action_preds, action_target, reduction='mean')
        if cfg.loss_type == 'mse_over_angle':
            loss = mse_over_angle(action_preds, action_target)
        elif cfg.loss_type == 'mean_norm_of_vec_diff':
            loss = mean_norm_of_vec_diff(action_preds, action_target, cfg)
        elif cfg.loss_type == 'cosine_similarity':
            loss = cosine_similarity(action_preds, action_target, cfg)
        elif cfg.loss_type == 'cos_diff':
            loss = cos_diff(action_preds, action_target)
        elif cfg.loss_type == 'mae_over_angle':
            loss = mae_over_angle(action_preds, action_target)
        elif cfg.loss_type == "cross_entropy":
            loss = cross_entropy(action_preds, action_target)
        else:
            raise NotImplementedError
        
        # losses += loss.item()
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))
        if count%log_interval == 0:
            wandb.log({f"in_eval/avg_val_loss vs log_intervalth update": avg_loss})
        count += 1

    # all_att_mats = extract_attention_scores(model)
    return avg_loss, None


def translate(model: torch.nn.Module, test_idx, test_set, tr_set_stats, cfg, earlybreak=10**8, 
                simulate_tgt_actions=False):
    model.eval()
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    count = 0           # keeps count of total episodes
    success_count = 0   # keeps count of successful episodes
    op_traj_dict_list = []
    total_timesteps = 0
    total_reward = 0
    states_mean, states_std = tr_set_stats
    states_mean = torch.from_numpy(states_mean).to(cfg.device)
    states_std = torch.from_numpy(states_std).to(cfg.device)

    _, dummy_actions, _, _, dummy_states, _, _,dummy_flow_dir,_ , _ = test_set[0]
    src_vec_dim = dummy_states.shape[-1]

    max_test_ep_len = cfg.context_len
    act_dim = 1
    state_dim = src_vec_dim
    target_token = None
    device = cfg.device
    
    with torch.no_grad():
        # for sample in range(len(test_set)):
        # timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx = test_set[sample]

        for timesteps, tgt, traj_mask, target_state, ground_truth_states, traj_len, idx, flow_dir, rzn, _ in test_dataloader:
            # if idx%100==0:
            #     print(idx)
            # translate_one_rzn = timer()
            # set up environment
            flow_dir = flow_dir[0]

            env = setup_env(flow_dir)

            op_traj_dict = {}
            reached_target = False

            env.reset()

            idx = idx[0].item() #initially idx = tensor([0])

            timesteps = timesteps.to(cfg.device)
            count += 1
            if count == earlybreak:
                break
          

            actions = torch.zeros((1, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((1, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((1, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)
            context_len = cfg.context_len
            episode_returns = 0
            running_state = np.array(env.reset()) # running state is txy


            for t in range(cfg.context_len):
                total_timesteps += 1

                # add state in placeholder and normalize
                # states[0, t] = torch.from_numpy(running_state).to(device)
                if state_dim == 5:
                    uv = env.get_velocity(running_state)  
                    running_state = np.append(running_state, uv) # now, running state is txyuv
                
                states[0, t] = torch.from_numpy(running_state).to(device)
                
                
                states[0, t] = (states[0, t] - states_mean) / states_std

                # calcualate running rtg and add it in placeholder
                rewards_to_go[0, t] = 0
                # start_fp = time.time()
                if simulate_tgt_actions:
                    act = tgt[0,t]
                else:
                    if t < context_len:
                        _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                    states[:,:context_len],
                                                    actions[:,:context_len],
                                                    rewards_to_go[:,:context_len],
                                                    target_token=target_token)
                        act = act_preds[0, t].detach()
                    else:
                        _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                    states[:,t-context_len+1:t+1],
                                                    actions[:,t-context_len+1:t+1],
                                                    rewards_to_go[:,t-context_len+1:t+1],
                                                    target_token=target_token)
                        act = act_preds[0, -1].detach()
           
                
                running_state, running_reward, done, info = env.step(act.cpu().numpy()*2*np.pi)
                

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward
                episode_returns += running_reward

                if done:
                    if t+1 < env.nT :
                        # states[0, t+1] = torch.from_numpy(running_state).to(device)
                        if state_dim == 5:
                            if env.is_outbound(running_state):
                                uv = (0, 0)
                            else:
                                uv = env.get_velocity(running_state)  
                            running_state = np.append(running_state, uv) # now, running state is txyuv
                        states[0, t+1] = torch.from_numpy(running_state).to(device)
                        states[0, t+1] = (states[0, t+1] - states_mean) / states_std
                    # if done and getting positive reward
                    if running_reward > 0: 
                        reached_target = True
                        success_count += 1
                    break
                # end_fp = time.time()
                # print(f"rebuttal: forward pass time: {end_fp - start_fp}")


            op_traj_dict['states'] = states.cpu()
            op_traj_dict['actions'] = actions.cpu()*2*np.pi
            op_traj_dict['t_done'] = t+1
            op_traj_dict['n_tsteps'] = t+2
            op_traj_dict['success'] = reached_target
            op_traj_dict['ground_truth_states'] = ground_truth_states
            op_traj_dict['flow_dir'] = flow_dir
            # op_traj_dict['mse'] = mse
            # op_traj_dict['all_att_mat'] = extract_attention_scores(model)
            # op_traj_dict['states_for_action_labels'] = None
            # op_traj_dict['action_labels'] = tgt.cpu()*2*np.pi

            # op_traj_dict['success_fal'] = reached_target_
            op_traj_dict_list.append(op_traj_dict)

    # mean_translate_time = np.mean(np.array(translate_one_rzn_list)[1:101])
    # print(f'Avg translate time for 100 rzns: {mean_translate_time}')
    # print(np.array(translate_one_rzn_list)[1:101].shape)
    results = {}
    # results['avg_val_loss'] = np.mean([d['mse'] for d in op_traj_dict_list])
    results['translate/avg_ep_len'] = np.mean([d['n_tsteps'] for d in op_traj_dict_list])
    results['translate/success_ratio'] = success_count/count
    results['runs_from_set(count)'] = count

    return op_traj_dict_list, results



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


    wandb_exp_name = "DT_rsa-a_" + dataset_name + "__" + start_time_str
    wandb.init(project="dt",
        name = wandb_exp_name,
        config=config
        )

    if not config: # For sweep mode -> since config is set to default ({ }) we have to fill it again from wandb
        wb_id = wandb.run.get_url()[-8:]
        api = wandb.Api()
        run = api.run(f"wandb-iisc/dt/runs/{wb_id}")
        config = run.config
        config['wb_id'] = wb_id
    
    cfg=wandb.config
    # params2 = read_cfg_file(cfg_name=join(ROOT,cfg.params2_name))

    lr = cfg.lr    
    num_epochs = cfg.num_epochs
    context_len = cfg.context_len     # K in decision transformer
    device = torch.device(cfg.device)

    # load data from this file
    dataset_path = join(ROOT,cfg.dataset_path)
    dataset_name = cfg.dataset_name
    # saves model and csv in this directory
    log_dir = join(ROOT, cfg.log_dir)

    if ARGS_QR:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        num_epochs = 2

    prefix = "my_dt" + dataset_name
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

    # env = gym.make(env_name)
    # env.setup(cfg, params2, add_trans_noise=add_trans_noise)

    # Load and Split dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    idx_split, set_split = get_data_split(traj_dataset,
                                        split_ratio=cfg.split_tr_tst_val, 
                                        random_seed=cfg.split_ran_seed, 
                                        random_split=cfg.random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split

    # Temp hard code to remove blue traj from train set
    # temp_list = list(train_traj_set)
    # del temp_list[3]
    # train_traj_set = tuple(temp_list)
    # del train_idx_set[3]

    try:
        if args.train_with_1_sample:
            # dataset contains optimal actions for different realizations of the env 
            tr_set = create_dt_dataset_v3(train_traj_set, 
                                    train_idx_set,
                                    context_len, 
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
                                                ) #tuple([train_traj_set[0]])

            src_stats = tr_set.get_src_stats()
            states_stats_path = join(model_save_dir,"states_stats.npy")
            np.save(states_stats_path, src_stats)
            
            val_set = create_dt_dataset_v3(tuple([val_traj_set]), 
                                    val_idx_set,
                                    context_len,
                                    norm_params_4_val = src_stats,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,

                                                )
            test_set = create_dt_dataset_v3(tuple([test_traj_set]), 
                                    val_idx_set,
                                    context_len, 
                                    norm_params_4_val = src_stats,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,

                                                )
        else:
            tr_set = create_dt_dataset_v3(train_traj_set, 
                                    train_idx_set,
                                    context_len,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
 
                                                )

            src_stats = tr_set.get_src_stats()
            states_stats_path = join(model_save_dir,"states_stats.npy")
            np.save(states_stats_path, src_stats)
            
            val_set = create_dt_dataset_v3(val_traj_set, 
                                    val_idx_set,
                                    context_len,
                                    norm_params_4_val = src_stats,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
                                                )
            test_set = create_dt_dataset_v3(test_traj_set, 
                                    val_idx_set,
                                    context_len, 
                                    norm_params_4_val = src_stats,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
                                                )
    except:
        if not args: 
            # dataset contains optimal actions for different realizations of the env 
            # TODO : Remove hardcode of 0th train set, done for verification purposes (Shubham)
            tr_set = create_dt_dataset_v3(train_traj_set, 
                                    train_idx_set,
                                    context_len, 
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
                                                ) #tuple([train_traj_set[0]])

            src_stats = tr_set.get_src_stats()
            states_stats_path = join(model_save_dir,"states_stats.npy")
            np.save(states_stats_path, src_stats)
            
            val_set = create_dt_dataset_v3(val_traj_set, 
                                    val_idx_set,
                                    context_len,
                                    norm_params_4_val = src_stats,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
                                                )
            test_set = create_dt_dataset_v3(test_traj_set, 
                                    val_idx_set,
                                    context_len, 
                                    norm_params_4_val = src_stats,
                                    use_txyuv_as_states=cfg.use_txyuv_as_states,
                                                )           

    _, dummy_actions, _, _, dummy_states, _, _,dummy_flow_dir,_,_ = tr_set[0]
    src_vec_dim = dummy_states.shape[-1]
    tgt_vec_dim = dummy_actions.shape[-1]
    print(f"src_vec_dim = {src_vec_dim} \n tgt_vec_dim = {tgt_vec_dim}")
    
    # intantiate gym env for vizualization purposes
    env_4_viz = setup_env(dummy_flow_dir)
    
    # # for Debugging
    break_at = 500

    # visualize_input(val_set, log_wandb=True, at_time=119, env=env_4_viz, break_at=break_at)
    # simulate_tgt_actions(val_set,
    #                         env=env_4_viz,
    #                         log_wandb=True,
    #                         wandb_fname='simulate_tgt_actions',
    #                         plot_flow=True,
    #                         at_time=119,
    #                         break_at=break_at)
    
    # sys.exit(0)
    
    # transformer = mySeq2SeqTransformer_v1(num_encoder_layers, num_decoder_layers, embed_dim,
    #                              n_heads, src_vec_dim, tgt_vec_dim, 
    #                              dim_feedforward=None,     # TODO: add dim_ffn to cfg
    #                              max_len=context_len,
    #                              positional_encoding="simple"
    #                              ).to(cfg.device)
    
    sigmoid_activation = True if cfg.action_activation == 'sigmoid' else False
    transformer = DecisionTransformer(
                state_dim=src_vec_dim,
                act_dim=tgt_vec_dim,
                n_blocks=cfg.num_encoder_layers,
                h_dim=cfg.embed_dim,
                context_len=context_len,
                n_heads=cfg.n_heads,
                drop_p=cfg.dropout_p,
                max_timestep=cfg.context_len,
                sigmoid_activation_in_action=sigmoid_activation,
                dont_use_rtg=cfg.dont_use_rtg_in_input_seq
                # target_token=cfg.target_conditioning    #True/False
                    ).to(device)
    
    if cfg.optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
                        transformer.parameters(),
                        lr=lr,
                        weight_decay=cfg.wt_decay
                    )
    elif cfg.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(transformer.parameters(), 
                                    lr=lr, 
                                    weight_decay=cfg.wt_decay,
                                    betas=(0.9, 0.98), 
                                    eps=1e-9, 
                                    )    
    # main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
    #                                                                T_max=num_epochs-3, 
    #                                                                eta_min = 0.0
    #                                                                 )
    gamma, _ = see_steplr_trend(step_size=10, num_epochs=num_epochs, lr=lr, final_lr=cfg.final_lr)
    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                          start_factor=0.033,
                                                          total_iters=3
                                                          )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                      schedulers=[warm_up_scheduler, main_lr_scheduler],
                                                      milestones=[3])

    pytorch_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

    min_ETA = 10**5
    max_sr = -1
    max_sr_train = -1
    min_train_loss = 10**5
    # val_loss = 0
    for epoch in range(0, num_epochs+1):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        print("training")
        train_loss, _ = train_epoch(transformer, optimizer, tr_set, cfg, args, scheduler=scheduler)
        epoch_end_time = timer()
        print("evaluating")
        val_loss, _ = evaluate(transformer, val_set, cfg)
        scheduler.step()
        wandb.log({f"in_eval/lr":  scheduler.get_last_lr()[0]
                       })
        
        # Evalutation by translation   
        if epoch % cfg.eval_inerval == 0:
            # print("plotting attention")
            # plot_all_attention_mats(tr_all_att_mat)
            # plot_all_attention_mats(val_all_att_mat)
            print("translating")
            # tr_op_traj_dict_list, tr_results = translate(transformer, train_idx_set, tr_set, None, 
            #                                        cfg, earlybreak=tt_eb[0])
            
            # tr_set_txy_preds = [d['states'] for d in tr_op_traj_dict_list]
            # all_att_mat_list =  [d['all_att_mat'] for d in tr_op_traj_dict_list]


            # path_lens = [d['n_tsteps'] for d in tr_op_traj_dict_list]
            # visualize_output(tr_set_txy_preds, 
            #                     path_lens,
            #                     iter_i = 0, 
            #                     stats=None, 
            #                     env=env_4_viz, 
            #                     log_wandb=True, 
            #                     plot_policy=False,
            #                     traj_idx=None,      #None = all, list of rzn_ids []
            #                     show_scatter=False,
            #                     at_time=None,
            #                     color_by_time=True, #TODO: fix tdone issue in src_utils
            #                     plot_flow=True,
            #                     wandb_suffix="train")
            
            # compare_trajectories(tr_op_traj_dict_list,
            #                     path_lens,
            #                     iter_i = 0, 
            #                     stats=None, 
            #                     env=env_4_viz, 
            #                     log_wandb=True, 
            #                     plot_policy=True,
            #                     traj_idx=[1, 5,],      #None=all, list of rzn_ids []
            #                     show_scatter=True,
            #                     at_time=None,
            #                     color_by_time=True, #TODO: fix tdone issue in src_utils
            #                     plot_flow=True,
            #                     wandb_suffix="train")   
            
      
            # viz_op_traj_with_attention(tr_set_txy_preds,
            #                     all_att_mat_list, # could be enc_sa, dec_sa, dec_ga
            #                     path_lens,
            #                     mode='dec_sa',       #or 'a_s_attention'
            #                     average_across_layers=True,
            #                     stats=None, 
            #                     env=env_4_viz, 
            #                     log_wandb=True, 
            #                     scale_each_row=True,
            #                     plot_policy=False,
            #                     traj_idx=None,      #None=all, list of rzn_ids []
            #                     show_scatter=False,
            #                     plot_flow=True,
            #                     at_time=88,
            #                     model_name="DOLS"+"_on_"+dataset_name
            #                     )  
                        
            train_op_traj_dict_list, train_results = translate(transformer, train_idx_set, tr_set, src_stats, 
                                                           cfg, earlybreak=cfg.translate_earlybreaks[2])
            
            train_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in train_op_traj_dict_list]
            path_lens = [d['n_tsteps'] for d in train_op_traj_dict_list]
            
            sim_train_op_traj_dict_list, sim_train_set_txy_preds= None, None
            sim_val_op_traj_dict_list, sim_val_set_txy_preds= None, None
            try:
                if args.train_with_1_sample or not args.train_with_1_sample:
                    sim_train_op_traj_dict_list, _ = translate(transformer, train_idx_set, tr_set, src_stats, 
                                                                cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                                simulate_tgt_actions=True)
                    sim_train_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in sim_train_op_traj_dict_list]  
                    ground_truth_states_train = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_train_op_traj_dict_list]     
                    sim_val_op_traj_dict_list, _ = translate(transformer, val_idx_set, val_set, src_stats, 
                                                                cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                                simulate_tgt_actions=True)
                    sim_val_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in sim_val_op_traj_dict_list] 
                    ground_truth_states_val = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_val_op_traj_dict_list]        
            except:
                if not args:
                    sim_train_op_traj_dict_list, _ = translate(transformer, train_idx_set, tr_set, src_stats, 
                                                                cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                                simulate_tgt_actions=True)
                    sim_train_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in sim_train_op_traj_dict_list]  
                    ground_truth_states_train = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_train_op_traj_dict_list]     
                    sim_val_op_traj_dict_list, _ = translate(transformer, val_idx_set, val_set, src_stats, 
                                                                cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                                simulate_tgt_actions=True)
                    sim_val_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in sim_val_op_traj_dict_list] 
                    ground_truth_states_val = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_val_op_traj_dict_list]        
                                      
            colours = visualize_output(train_set_txy_preds, 
                                path_lens,
                                iter_i = 0, 
                                stats=None, 
                                env=env_4_viz, 
                                log_wandb=True, 
                                plot_policy=False,
                                traj_idx=None,      #None=all, list of rzn_ids []
                                show_scatter=True,
                                at_time=59,
                                color_by_time=True, #TODO: fix tdone issue in src_utils
                                plot_flow=True,
                                wandb_suffix="train",
                                ground_truth_states = ground_truth_states_train, #for debugging or comparison
                                preds_list_from_target_actions = sim_train_set_txy_preds, # for debugging or comparison
                                ) 
            
            val_op_traj_dict_list, val_results = translate(transformer, val_idx_set, val_set, src_stats, 
                                                           cfg, earlybreak=cfg.translate_earlybreaks[2])
            val_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in val_op_traj_dict_list]
            path_lens = [d['n_tsteps'] for d in val_op_traj_dict_list]
            visualize_output(val_set_txy_preds, 
                                path_lens,
                                iter_i = 0, 
                                stats=None, 
                                env=env_4_viz, 
                                log_wandb=True, 
                                plot_policy=False,
                                traj_idx=None,      #None=all, list of rzn_ids []
                                show_scatter=True,
                                at_time=59,
                                color_by_time=True, #TODO: fix tdone issue in src_utils
                                plot_flow=True,
                                wandb_suffix="val",
                                ground_truth_states = ground_truth_states_val,
                                preds_list_from_target_actions = sim_val_set_txy_preds,
                                )# for debugging or comparison) 

        #Note: val_results get updated after eval_inerval
        translate_avg_ep_len = val_results['translate/avg_ep_len']
        # translate_avg_val_loss = val_results['avg_val_loss']
        success_ratio = val_results['translate/success_ratio']
        train_success_ratio = train_results['translate/success_ratio']
        count = val_results['runs_from_set(count)'] 
        log_dict = { "tr_loss_vs_epoch (unpadded elems)": train_loss,
            "val_loss_vs_epoch (unpadded elems)": val_loss,
            # "avg_val_loss (across pred len)": translate_avg_val_loss,
            "success_ratio": success_ratio,
            "succes_ratio (train)": train_success_ratio,
            'runs_from_set(count)': count,
            "ETA": translate_avg_ep_len,
            # "lr" : scheduler.get_last_lr()[0] if use_scheduler else lr
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
 
        
        # TODO: save model and best metrics after completing evaluation
        # if translate_avg_ep_len < min_ETA:
        if success_ratio > max_sr:
            # min_ETA = translate_avg_ep_len
            max_sr = success_ratio
            print("saving current model based on val success ratio at: " + model_save_dir)

            best_avg_episode_length = translate_avg_ep_len
            best_success_ratio = success_ratio
            best_epoch = epoch
            # "avg_val_loss"= eval_avg_val_loss

            torch.save(transformer, join(model_save_dir, "best_sr_val.pt"))
        
        if train_success_ratio > max_sr_train:
            # min_ETA = translate_avg_ep_len
            max_sr_train = train_success_ratio
            print("saving current model based on train success ratio at: " + model_save_dir)
            torch.save(transformer, join(model_save_dir, "best_sr_train.pt"))
     
        
    print("saving model after laste epoch: " + model_save_dir)
    torch.save(transformer, join(model_save_dir, "min_train_loss.pt"))

    cfg_copy_path = model_save_dir[:-2] + "yml"
    save_yaml(cfg_copy_path,cfg)
    print(f"cfg_copy_path = {cfg_copy_path}")
    wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    wandb.run.summary["best_success_ratio"] = best_success_ratio
    wandb.run.summary["total_runs_val_set"] = len(val_op_traj_dict_list)
    wandb.run.summary["best_epoch"] = best_epoch

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    tmp_path = join(model_save_dir, "min_train_loss.pt")
    print(f"\n\n ---- running inference on test set ----- \n\n")
    transformer = torch.load(tmp_path)
    
    train_op_traj_dict_list, train_results = translate(transformer, train_idx_set, tr_set, src_stats, 
                                                    cfg, earlybreak=cfg.translate_earlybreaks[2])
    
    train_set_actions_preds = [d['actions'] for d in train_op_traj_dict_list]
    
    sim_train_op_traj_dict_list, sim_train_set_txy_preds= None, None
 
    try:
        if args.train_with_1_sample or not args.train_with_1_sample:
            sim_train_op_traj_dict_list, _ = translate(transformer, train_idx_set, tr_set, src_stats, 
                                                        cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                        simulate_tgt_actions=True)
            train_set_actions_ground = [d['actions'] for d in sim_train_op_traj_dict_list]    
     
    except:
        if not args:
            sim_train_op_traj_dict_list, _ = translate(transformer, train_idx_set, tr_set, src_stats, 
                                                        cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                        simulate_tgt_actions=True)
            train_set_actions_ground = [d['actions'] for d in sim_train_op_traj_dict_list]    
            
    flow_dir = [d['flow_dir'] for d in train_op_traj_dict_list]
    visualize_actions(model_save_dir,
                    flow_dir, 
                    colours, 
                    train_set_actions_preds, 
                    train_set_actions_ground,
                    log_wandb=True, 
                    )

    
    op_traj_dict_list, results  = translate(transformer,test_idx_set, test_set, 
                                            src_stats, cfg, earlybreak=cfg.translate_earlybreaks[1])
    test_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in op_traj_dict_list]
    path_lens = [d['n_tsteps'] for d in op_traj_dict_list]  
    sim_test_op_traj_dict_list, sim_test_set_txy_preds= None, None
    try:
        if args.train_with_1_sample or not args.train_with_1_sample:
        
            sim_test_op_traj_dict_list, _ = translate(transformer, test_idx_set, test_set, src_stats, 
                                                        cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                        simulate_tgt_actions=True)
            sim_test_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in sim_test_op_traj_dict_list]  
            ground_truth_states_test = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_test_op_traj_dict_list] 
    except:
        if not args:
            sim_test_op_traj_dict_list, _ = translate(transformer, test_idx_set, test_set, src_stats, 
                                                        cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                        simulate_tgt_actions=True)
            sim_test_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in sim_test_op_traj_dict_list]  
            ground_truth_states_test = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_test_op_traj_dict_list]
            
        
    visualize_output(test_set_txy_preds, 
                        path_lens,
                        iter_i = 0, 
                        stats=None, 
                        env=env_4_viz, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=True,
                        at_time=None,
                        color_by_time=True, #TODO: fix tdone issue in src_utils
                        plot_flow=True,
                        wandb_suffix="test",
                        ground_truth_states=ground_truth_states_test, #for debugging or comparison
                        preds_list_from_target_actions=sim_test_set_txy_preds) # for debugging or comparison)
    
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved last updated model at: " + model_save_dir)
    print("=" * 60)

    return best_avg_episode_length

class jugaad_cfg:
    def __init__(self, context_len, device):
        self.context_len = context_len
        self.device = device

def load_prev_and_test(args, cfg_name):

    tmp_path = join(args.log_exp_dir, "best_sr_train.pt")

    cfg_path = join(args.log_exp_dir, "run_cfg.yml")
    cfg = read_cfg_file(cfg_path)
    cfg = convert_dict_to_obj(cfg)

    transformer = torch.load(tmp_path)
    model_name = tmp_path[:-3].split('/')[-1]

    dset = 'train'
    # dataset_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_multi_ran_stat/Gathered_datasets/gathered_10_0_5k_test/gathered_10.pkl"
    dataset_path = cfg.dataset_path
    traj_dataset = load_pkl(dataset_path)
    dataset_name = dataset_path[:-4].split('/')[-1]
    # src_stats_path = tmp_path[:-3] + "_src_stats.npy"
    src_stats_path = join(args.log_exp_dir, "states_stats.npy")
    src_stats = np.load(src_stats_path)
    src_stats = (src_stats[0], src_stats[1])

    idx_split, set_split = get_data_split(traj_dataset,
                                    split_ratio=cfg.split_tr_tst_val, 
                                    random_seed=cfg.split_ran_seed,
                                    random_split=cfg.random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split
    
    train_traj_set = create_dt_dataset_v3(train_traj_set, 
                            idx_set=[None],
                            context_len=cfg.context_len,
                            # norm_params_4_val = src_stats
                                        )
    
    val_traj_set = create_dt_dataset_v3(val_traj_set, 
                            idx_set=[None],
                            context_len=cfg.context_len,
                            norm_params_4_val = src_stats
                                        )
    test_traj_set = create_dt_dataset_v3(test_traj_set, 
                            idx_set=[None],
                            context_len=cfg.context_len,
                            norm_params_4_val = src_stats
                                        )
    
    # src_stats = us_test_traj_set.get_src_stats()
    # test_idx_set = None #TODO: clean unneeded vars and args
    # translate_start_time = timer()

    train_op_traj_dict_list, train_results = translate(transformer,train_idx_set, train_traj_set, 
                                            src_stats, cfg, earlybreak=cfg.translate_earlybreaks[2])
    train_set_actions_preds = [d['actions'] for d in train_op_traj_dict_list]
    sim_train_op_traj_dict_list, _ = translate(transformer, train_idx_set, train_traj_set, src_stats, 
                                                cfg, earlybreak=cfg.translate_earlybreaks[2],
                                                simulate_tgt_actions=True)
    train_set_actions_ground = [d['actions'] for d in sim_train_op_traj_dict_list]  
    visualize_actions_flow(args.log_exp_dir, 
                           train_op_traj_dict_list, 
                           sim_train_op_traj_dict_list, 
                           src_stats=src_stats, 
                           log_wandb=False)
    # translate_end_time = timer()
    # print(f"Translate runtime = {(translate_end_time - translate_start_time):.3f}s")
    os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"),exist_ok=True)
    os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"), exist_ok=True)
    save_object(op_traj_dict_list, os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"))
    save_object(results,os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"))

    op_traj_dict_list = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"))
    results = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"))
    _, dummy_actions, _, _, dummy_states, _, _,dummy_flow_dir,_ , _ , _= us_train_traj_set[0]
    # _, dummy_target, _, _, dummy_env_coef_seq, _,_,dummy_flow_dir,_ = us_val_traj_set[0]
    # intantiate gym env for vizualization purposes
    env_4_viz = setup_env(dummy_flow_dir)

    test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
    path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
    # all_att_mat_list =  [d['all_att_mat'] for d in op_traj_dict_list]
    success_list = [d['success'] for d in op_traj_dict_list]
    actions = [d['actions'] for d in op_traj_dict_list]
    
    # taken from vis_traj_with_attention.py in decision transformer project
    print(f"model_name = {model_name}")
    save_dir = "paper_plots/"  + model_name + f"/{dataset_name}/dt_bc"
    save_dir = join(ROOT,save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    paper_plot_info = {"trajs_by_arr": {"fname":"T_arr"},
                    "trajs_by_att": {"ts":[17,46, 70],"fname":"att"},
                    "att_heatmap":{"fname":"heatmap"},
                    "plot_val_ip_op":{"fname":"test_ip_op"},
                    "plot_trajs_ip_op":{"fname":"test_trajs_ip_op"},
                    "plot_actions":{"fname":"test_actions_ip_op"},
                    "plot_train_val_ip_op":{"fname":"plot_train_val_ip_op"},
                    "loss_avg_returns":{"fname":"loss"},
                    "vel_field":{"fname":"vel_field", "ts":[1,60,119]}
                        }
    pp = paper_plots(env_4_viz, op_traj_dict_list, src_stats,
                        paper_plot_info=paper_plot_info,
                        save_dir=save_dir)
    pp.plot_val_ip_op(us_test_traj_set, test_set_txy_preds, path_lens, success_list)
    # pp.plot_trajs_ip_op(us_test_traj_set, test_set_txy_preds, path_lens, success_list)
    # pp.plot_velocity_obstacle()
    # pp.plot_actions(us_test_traj_set, test_set_txy_preds, path_lens, success_list, actions)
    
    # pp.plot_traj_by_arr(us_test_traj_set,set_str="_us_test_")
    # pp.plot_train_val_ip_op(us_train_traj_set, us_test_traj_set)
    # pp.plot_traj_by_arr(val_traj_dataset, set_str="_val")
    # pp.plot_att_heatmap(100)
    # pp.plot_traj_by_att("a_a_attention")
    # pp.plot_traj_by_att("a_s_attention")
    
    #  visualize_output(test_set_txy_preds, 
    #                     path_lens,
    #                     iter_i = 0, 
    #                     stats=None, 
    #                     env=env_4_viz, 
    #                     log_wandb=False, 
    #                     plot_policy=False,
    #                     traj_idx=None,      #None=all, list of rzn_ids []
    #                     show_scatter=False,
    #                     at_time=None,
    #                     color_by_time=True, #TODO: fix tdone issue in src_utils
    #                     plot_flow=True,
    #                     wandb_suffix="test_on_unseen",
    #                     model_name=model_name+"_on_"+dataset_name)
    
    # plot_all_attention_mats(all_att_mat_list[0],
    #                         log_wandb=False, 
    #                         model_name=model_name+"_on_"+dataset_name)
    # for t in [i*10 for i in range(1,11)]:
    #     viz_op_traj_with_attention(test_set_txy_preds,
    #                         all_att_mat_list, 
    #                         path_lens,
    #                         mode='dec_sa',       #or 'a_s_attention'
    #                         average_across_layers=True,
    #                         stats=None, 
    #                         env=env_4_viz, 
    #                         log_wandb=False, 
    #                         scale_each_row=True,
    #                         plot_policy=False,
    #                         traj_idx=None,      #None=all, list of rzn_ids []
    #                         show_scatter=False,
    #                         plot_flow=True,
    #                         at_time=t,
    #                         model_name=model_name+"_on_"+dataset_name
    #                         )
    # viz_op_traj_with_attention(test_set_txy_preds,
    #                     all_att_mat_list, 
    #                     path_lens,
    #                     mode='dec_ga',       #or 'a_s_attention'
    #                     average_across_layers=True,
    #                     stats=None, 
    #                     env=env_4_viz, 
    #                     log_wandb=False, 
    #                     scale_each_row=True,
    #                     plot_policy=False,
    #                     traj_idx=None,      #None=all, list of rzn_ids []
    #                     show_scatter=False,
    #                     plot_flow=True,
    #                     at_time=None,
    #                     model_name=model_name+"_on_"+dataset_name
    #                     )   
    # print(f" Results on unseen test")
    # print_dict(results)  
    # # visualize_input(us_test_traj_set, at_time=119, 
    # #                                   env=env_4_viz,
    # #                                   log_wandb=False,
    # #                                   data_name=dataset_name
    # #                                         )

    # print(f" Results on unseen test")
    # print_dict(results)
    # return        


# train_op_traj_dict_list, train_results = translate(transformer,train_idx_set, train_traj_set, 
#                                         src_stats, cfg, earlybreak=cfg.translate_earlybreaks[2])
# train_set_actions_preds = [d['actions'] for d in sim_train_op_traj_dict_list]
# sim_train_op_traj_dict_list, _ = translate(transformer, train_idx_set, train_traj_set, src_stats, 
#                                             cfg, earlybreak=cfg.translate_earlybreaks[2],
#                                             simulate_tgt_actions=True)
# train_set_actions_ground = [d['actions'] for d in sim_train_op_traj_dict_list]  

# plt.cla()
# plt.plot(train_set_actions_ground[0][0], color='green', label='ground_truth_actions')
# plt.plot(train_set_actions_preds[0][0], linestyle='dotted', color='red', label='preds_actions')
# plt.legend(loc='upper right', fontsize='small')
# plt.savefig(join(args.log_exp_dir, 'actions_plot.png'), dpi=300)



ARGS_QR = False
ARGS_CFG = "_--_"
NAME_MAP = {
            "v5": "DG3",
            "v5_HW": "HW",
            "v5_GPT_DG3": "GPT_DG3",
            "v5_DOLS": "DOLS",
            "v5_Perlin": "Perlin",
            'v5_Perlin_pptt': "Perlin"
            }

if __name__ == "__main__":

    # def_log_exp_dir = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/log/my_dtPerlin_model_04-29-16-55-54"
    # def_log_exp_dir = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/log/my_dtPerlin_model_04-29-10-32-48"
    # def_log_exp_dir = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/log/my_dtPerlin_model_05-02-16-16-31"
    def_log_exp_dir = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/log/my_dtPerlin_model_05-08-11-52-47"

    print(f"cuda available: {torch.cuda.is_available()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_exp_dir', type=str, default=def_log_exp_dir)
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    parser.add_argument('--CFG', type=str, default='v5_Perlin_pptt')
    parser.add_argument('--train_with_1_sample', type=bool, default=False)
    args = parser.parse_args()

    cfg_name = "cfg/contGrid_" + args.CFG
    sweep_cfg_name = cfg_name + "_sweep_dt"
    cfg_name =  cfg_name + ".yaml"
    sweep_cfg_name =  sweep_cfg_name + ".yaml"
    cfg_name = join(ROOT,cfg_name)
    sweep_cfg_name = join(ROOT,sweep_cfg_name)

    ARGS_QR = args.quick_run # for sweep mode
    ARGS_CFG = args.CFG
    print(f'args.mode = {args.mode}')
    print(f"ARGS_QR={ARGS_QR}")

    if args.mode == 'load_and_translate':
        print("----- beginning load_prev_and_test ------")
        load_prev_and_test(args, cfg_name)        

    if args.mode == 'single_run':
        print("----- beginning single_run ------")
        train_model(args, cfg_name)

    elif args.mode == 'sweep':
        print("----- beginning sweeeeeeeeeeeeep ------")
        sweep_cfg = read_cfg_file(cfg_name=sweep_cfg_name)
        sweep_id = wandb.sweep(sweep_cfg)
        wandb.agent(sweep_id, function=train_model)