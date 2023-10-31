import torch
from model_min_dt import Transformer_decoder
import pickle
from src_utils import create_env_emb_dataset, get_data_split, see_steplr_trend
from utils import read_cfg_file, load_pkl, print_dict, save_object
from torch.utils.data import DataLoader
import numpy as np
import wandb
from torch.nn import functional as F
from datetime import datetime
import argparse
from os.path import join
from root_path import ROOT
from timeit import default_timer as timer


wandb.login()

def train_epoch(model, optimizer, train_dataloader, cfg, scheduler=None, log_interval=50):
    model.train()
    avg_loss = 0

    loss = 0
    count = 0
    # for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, _, _ in train_dataloader:
    for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in train_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_reps[:, :-1, :].to(cfg.device)
        tgt = env_reps[:, 1:, :].to(cfg.device)

        # src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(timesteps, src)

        loss = F.mse_loss(logits, tgt)
        optimizer.zero_grad()

        loss.backward()
        # TODO: try 10
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4)

        optimizer.step()
        # TODO: See if scheduler can be applied at update level
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))

        if count%log_interval == 0:
            param_norms = [p.grad.data.norm(2).item() for p in model.parameters()]
            mean_norm = np.mean(param_norms)
            min_norm = np.min(param_norms)
            max_norm = np.max(param_norms)
            wandb.log({f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
                        "in_eval/MAX_param_norm": max_norm,
                        "in_eval/MIN_param_norm": min_norm,
                        "in_eval/AVG_param_norm": mean_norm,
                       })
        count += 1

    # all_att_mats = extract_attention_scores(model)
    all_att_mats = None
    return avg_loss, all_att_mats

def evaluate(model, val_dataloader, cfg, scheduler=None, log_interval=50):
    model.eval()
    avg_loss = 0

    loss = 0
    count = 0
    # for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, _, _ in train_dataloader:
    for timesteps, env_reps, traj_mask, idx, flow_dir, rzn in val_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_reps[:, :-1, :].to(cfg.device)
        tgt = env_reps[:, 1:, :].to(cfg.device)

        # src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(timesteps, src)

        loss = F.mse_loss(logits, tgt)

        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))

        if count%log_interval == 0:
            wandb.log({f"in_eval/avg_val_loss vs log_intervalth update": avg_loss})
        count += 1

    # all_att_mats = extract_attention_scores(model)
    all_att_mats = None
    return avg_loss, all_att_mats


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
           
            dyn_inputs[0,0,:] = src[0,0,:]
 
            loss_post_t = []
            for i in range(1, cfg.context_len-1):
                dyn_inputs[0,0:i,:] = src[0,0:i,:]
                preds = model(timesteps, dyn_inputs)
                loss_post_t.append(F.mse_loss(preds[0,i:,:],tgt[0,i:,:]))

            full_time_mse = F.mse_loss(preds[0,:],tgt[0,:])
            full_time_mse_list.append(full_time_mse)
            sample_results['loss_post_t'] = loss_post_t
            sample_results['full_time_mse'] = full_time_mse
            set_output.append(sample_results)

    avg_results = {}
    avg_results['avg_full_time_mse'] = np.mean(full_time_mse_list)

    return set_output, avg_results


def train_model(args=None, cfg_name=None):
    

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")
    
    
    cfg = read_cfg_file(cfg_name=cfg_name)

    dataset_name = cfg['dataset_name']
    wandb_exp_name = "emb_env_td_" + dataset_name + "__" + start_time_str
    wandb.init(project="td_emb",
                name = wandb_exp_name,
                config = cfg    
                )

    cfg=wandb.config

    drop_p = cfg.drop_p
    n_heads = cfg.n_heads
    context_len = cfg.context_len
    h_dim = cfg.h_dim
    n_blocks = cfg.n_blocks
    bs = cfg.bs
    dataset_path = cfg.dataset_path
    B = cfg.B
    T = cfg.T
    lr = cfg.lr
    wt_decay = cfg.wt_decay
    log_dir = cfg.log_dir
    num_epochs = cfg.num_epochs
    final_lr = cfg.final_lr
    tt_eb = cfg.translate_earlybreaks
    eval_inerval =  cfg.eval_inerval
    
    device = torch.device(cfg.device)
    torch.cuda.empty_cache()

    prefix = "my_emb_env_td_" + dataset_name

    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    save_model_path = join(log_dir, save_model_name)


    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    
    # Load and Split dataset
    emb_dataset = load_pkl(dataset_path)
    idx_split, set_split = get_data_split(emb_dataset,
                                        split_ratio=cfg.split_tr_tst_val, 
                                        random_seed=cfg.split_ran_seed, 
                                        random_split=cfg.random_split)
    train_emb_set, test_emb_set, val_emb_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split
    tr_set = create_env_emb_dataset(train_emb_set, train_idx_set, context_len)
    val_set = create_env_emb_dataset(val_emb_set, val_idx_set, context_len)
    test_set = create_env_emb_dataset(test_emb_set, test_idx_set, context_len)
    tr_dataloader = DataLoader(tr_set, batch_size=bs, shuffle=True,
                                    #   num_workers=10, pin_memory=True
                                    )
    val_dataloader = DataLoader(val_set, batch_size=bs, shuffle=True,
                                    #   num_workers=10, pin_memory=True
                                    )
    test_dataloader = DataLoader(test_set, batch_size=bs, shuffle=True,
                                    #   num_workers=10, pin_memory=True
                                    )

    timesteps, env_reps, traj_mask, idx, flow_dir, rzn = tr_set[0]
    src_vec_dim = env_reps.shape[-1] 
    print(f"src_vec_dim = {src_vec_dim}")

    
    transformer = Transformer_decoder(src_vec_dim, n_blocks, h_dim, context_len, n_heads, drop_p).to(cfg.device)
    optimizer = torch.optim.AdamW(
                    transformer.parameters(),
                    lr=lr,
                    weight_decay=wt_decay
                )
    
 
    
    # if optimizer_name == 'AdamW':
    #     optimizer = torch.optim.AdamW(
    #                     transformer.parameters(),
    #                     lr=lr,
    #                     weight_decay=wt_decay
    #                 )
    # elif optimizer_name == 'Adam':
    #     optimizer = torch.optim.Adam(transformer.parameters(), 
    #                                 lr=lr, 
    #                                 weight_decay=wt_decay,
    #                                 betas=(0.9, 0.98), 
    #                                 eps=1e-9, 
    #                                 )    
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

    pytorch_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params



    min_ETA = 10**5
    max_sr = -1
    max_avg = 10000
    # train_loss = 0
    # val_loss = 0
    for epoch in range(0, num_epochs+1):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        print("training")
        train_loss, tr_all_att_mat = train_epoch(transformer, optimizer, tr_dataloader, cfg, scheduler=scheduler)
        epoch_end_time = timer()
        print("evaluating")
        val_loss, val_all_att_mat = evaluate(transformer, val_dataloader, cfg)
        scheduler.step()
        wandb.log({f"in_eval/lr":  scheduler.get_last_lr()[0]
                       })
        
        # Evaluation by translation   
        if epoch % eval_inerval == 0:

            print("translating")
            tr_set_output, tr_avg_results = translate(transformer, train_idx_set, tr_set, None, 
                                                   cfg, earlybreak=tt_eb[0])
                
            val_set_output, val_avg_results = translate(transformer, val_idx_set, tr_set, None, 
                                                           cfg, earlybreak=tt_eb[1])


        # Note: val_results get updated after eval_inerval
        tr_avg_full_time_mse = tr_avg_results['avg_full_time_mse']
        val_avg_full_time_mse = val_avg_results['avg_full_time_mse']
        log_dict = { "tr_avg_full_time_mse": tr_avg_full_time_mse,
            "val_loss_vs_epoch (unpadded elems)": val_avg_full_time_mse,
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
        
        if val_avg_full_time_mse < max_avg:
            max_avg = val_avg_full_time_mse
            print("saving current model at: " + save_model_path)
            
            torch.save(transformer, save_model_path)
            tmp_path = save_model_path[:-1]
            torch.save(transformer, tmp_path)
            
    
        
    #     # TODO: save model and best metrics after completing evaluation
    #     # if translate_avg_ep_len < min_ETA:
    #     if success_ratio > max_sr:
    #         # min_ETA = translate_avg_ep_len
    #         max_sr = success_ratio
    #         print("saving current model at: " + save_model_path)

    #         best_avg_episode_length = translate_avg_ep_len
    #         best_success_ratio = success_ratio
    #         best_epoch = epoch
    #         # "avg_val_loss"= eval_avg_val_loss

    #         torch.save(transformer, save_model_path)
    #         tmp_path = save_model_path[:-1]
    #         torch.save(transformer, tmp_path)


    # cfg_copy_path = save_model_path[:-2] + "yml"
    # save_yaml(cfg_copy_path,cfg)
    # print(f"cfg_copy_path = {cfg_copy_path}")
    # wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    # wandb.run.summary["best_success_ratio"] = best_success_ratio
    # wandb.run.summary["total_runs_val_set"] = len(val_op_traj_dict_list)
    # wandb.run.summary["best_epoch"] = best_epoch

    # print("=" * 60)
    # print("finished training!")
    # print("=" * 60)

    # print(f"\n\n ---- running inference on test set ----- \n\n")
    # transformer = torch.load(tmp_path)
    # op_traj_dict_list, results  = translate(transformer,test_idx_set, test_set, 
    #                                         None, cfg, earlybreak=tt_eb[2])
    
    # end_time = datetime.now().replace(microsecond=0)
    # end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    # print("started training at: " + start_time_str)
    # print("finished training at: " + end_time_str)
    # print("total training time: " + time_elapsed)
    # print("saved last updated model at: " + save_model_path)
    # print("=" * 60)

    return 

# start_time = datetime.now().replace(microsecond=0)
# start_time_str = start_time.strftime("%m-%d-%H-%M")

# cfg_name = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/cfg/test_decoder.yaml"
# cfg = read_cfg_file(cfg_name=cfg_name)

# dataset_name = cfg['dataset_name']
# wandb_exp_name = "emb_env_td" + dataset_name + "__" + start_time_str
# wandb.init(project="td_emb",
#             name = wandb_exp_name,
#             config = cfg
            
#             )

# cfg = wandb.config

# drop_p = cfg.drop_p
# n_heads = cfg.n_heads
# context_len = cfg.context_len
# h_dim = cfg.h_dim
# n_blocks = cfg.n_blocks
# bs = cfg.bs
# dataset_path = cfg.dataset_path

# emb_dataset = load_pkl(dataset_path)
# idx_split, set_split = get_data_split(emb_dataset,
#                                     split_ratio=cfg.split_tr_tst_val, 
#                                     random_seed=cfg.split_ran_seed, 
#                                     random_split=cfg.random_split)
# train_emb_set, test_emb_set, val_emb_set = set_split
# train_idx_set, test_idx_set, val_idx_set = idx_split
# tr_set = create_env_emb_dataset(train_emb_set, train_idx_set, context_len)
# val_set = create_env_emb_dataset(val_emb_set, val_idx_set, context_len)
# tr_dataloader = DataLoader(tr_set, batch_size=bs, shuffle=True,
#                                 #   num_workers=10, pin_memory=True
#                                   )
# val_dataloader = DataLoader(val_set, batch_size=bs, shuffle=True,
#                                 #   num_workers=10, pin_memory=True
#                                   )
# dummy_env_emb = tr_set[0][1]
# src_vec_dim = dummy_env_emb.shape[-1] 
# # tgt_vec_dim = dummy_target.shape[-1]

# B = cfg.B
# T = cfg.T
# lr = cfg.lr
# wt_decay = cfg.wt_decay
# td = Transformer_decoder(src_vec_dim, n_blocks, h_dim, context_len, n_heads, drop_p).to(cfg.device)
# optimizer = torch.optim.AdamW(
#                 td.parameters(),
#                 lr=lr,
#                 weight_decay=wt_decay
#             )

# # data =  torch.rand((B,T,D))
# # timesteps = torch.arange(0,T)

# train_epoch(td, optimizer, tr_dataloader, cfg, )
# evaluate(td, val_dataloader, cfg, )
# print()



if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    parser.add_argument('--CFG', type=str, default='test_decoder')
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