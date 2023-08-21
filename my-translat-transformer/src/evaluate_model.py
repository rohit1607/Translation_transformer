import torch
from torch.utils.data import DataLoader
from src_utils import setup_env, create_mask

import numpy as np


class jugaad_cfg:
    def __init__(self, context_len, device):
        self.context_len = context_len
        self.device = device
        
        
# TODO: remove test_idx and tr_set_stats from args later for cleanup
def translate(model: torch.nn.Module, test_idx, test_set, tr_set_stats, cfg, earlybreak=10**8):
    model.eval()
    count = 0           # keeps count of total episodes
    success_count = 0   # keeps count of successful episodes
    op_traj_dict_list = []
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    with torch.no_grad():
        # for sample in range(len(test_set)):
        for timesteps, tgt, traj_mask, loc_tensor, vel_fld_seq, traj_len, idx, flow_dir, rzn in test_dataloader:
            if idx%100==0:
                print("in translate, idx=", idx)
            # set up environment
            # flow_dir = flow_dir[0]  # TODO: Verify (Shubham)
            
            env = setup_env(flow_dir[0])

            op_traj_dict = {}
            reached_target = False
            reached_target_ = False

            env.reset()

            # idx = idx[0].item() #initially idx = tensor([0]) # TODO: Verify (Shubham)
            # rzn = test_idx[idx]
            env.set_rzn(rzn)

            timesteps = timesteps.to(cfg.device)
            count += 1
            if count == earlybreak:
                break
            src = src = vel_fld_seq[:,0:cfg.context_len - 1].to(cfg.device) #vel_fld_seq.to(cfg.device) 
            dummy_tgt_for_mask = tgt.to(cfg.device)[:, :-1, :]
            loc_tensor = loc_tensor.to(cfg.device)

            src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, dummy_tgt_for_mask, traj_len, cfg.device, extra_tokens=1)
            # memory is the encoder output
            memory =  model.module.encode(src, loc_tensor, src_mask, timesteps)

            preds = torch.zeros((1, cfg.context_len, dummy_tgt_for_mask.shape[2]),dtype=torch.float32, device=cfg.device)
            # PREDS_ = torch.zeros((1, cfg.context_len, dummy_tgt_for_mask.shape[2]),dtype=torch.float32, device=cfg.device)
            
            txy_preds = np.zeros((1, cfg.context_len+1, 3),dtype=np.float32,)

            txy_preds[0,0,:] = np.array([0,env.start_pos[0],env.start_pos[1]])

            preds[0,0,:] = tgt[0,0,:]
            a = preds[0,0,:].cpu().numpy().copy()
            a = a*2*np.pi
            txy, reward ,done, info = env.step(a)
            txy_preds[0,1,:] = txy     

            for i in range(cfg.context_len-1):
                memory = memory.to(cfg.device)
                out = model.module.decode(preds, memory, tgt_mask, timesteps)
                gen = model.module.generator(out)
                preds[0,i+1,:] = gen[0,i,:].detach()
                a = preds[0,i+1,:].cpu().numpy().copy()
                a = a*2*np.pi
                txy, reward ,done, info = env.step(a)
                txy_preds[0,i+2,:] = txy 
                # TODO: ***IMP*****: reduce GPU-CPU communication
                if done:
                    if reward > 0:
                        reached_target = True
                        success_count += 1
                    break
            
         
            k = 0
            # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            mse = F.mse_loss(preds[0,:i].cpu(),tgt[0,:i].cpu())
            # print(f"rough mse for sample {count} = {mse}")

            op_traj_dict['states'] = np.array(txy_preds)
            op_traj_dict['actions'] = preds.cpu()*2*np.pi
            op_traj_dict['t_done'] = i+3
            op_traj_dict['n_tsteps'] = i+2
            # op_traj_dict['attention_weights'] = attention_weights
            op_traj_dict['success'] = reached_target
            op_traj_dict['mse'] = mse
            op_traj_dict['all_att_mat'] = extract_attention_scores(model.module)
            # op_traj_dict['states_for_action_labels'] = np.array(TXY_PREDS_)
            op_traj_dict['states_for_action_labels'] = None
            op_traj_dict['action_labels'] = tgt.cpu()*2*np.pi
            op_traj_dict['t_done_fal'] = k+1
            op_traj_dict['n_tsteps_fal'] = k
            # op_traj_dict['attention_weights'] = attention_weights
            op_traj_dict['success_fal'] = reached_target_
            op_traj_dict_list.append(op_traj_dict)

    results = {}
    results['avg_val_loss'] = np.mean([d['mse'] for d in op_traj_dict_list])
    results['translate/avg_ep_len'] = np.mean([d['n_tsteps'] for d in op_traj_dict_list])
    results['translate/success_ratio'] = success_count/count
    results['runs_from_set(count)'] = count
    return op_traj_dict_list, results



# 07-26-17-21
def load_prev_and_test(args, cfg_name):
    # load model
    # tmp_path = ROOT + "log/my_translat_GPTdset_DG3_model_04-01-03-20.pt"
    tmp_path = ROOT + "/log/my_translat_GenHW_model_08-06-18-19.pt"

    transformer = torch.load(tmp_path)
    model_name = tmp_path[:-3].split('/')[-1]
    dset = 'train'
    dataset_path = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GenHW_all/Gathered_datasets/gathered_16_500/gathered_16.pkl"
    traj_dataset = load_pkl(dataset_path)
    dataset_name = dataset_path[:-4].split('/')[-1]
    src_stats_path = tmp_path[:-3] + "_src_stats.npy"
    src_stats_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/log/my_translat_DOLS_Cylinder_model_05-19-14-08_src_stats.npy"
    src_stats = np.load(src_stats_path)
    src_stats = (src_stats[0], src_stats[1])


    traj_dataset =  traj_dataset[0:200]
    idx_split, set_split = get_data_split(traj_dataset,
                                    split_ratio=[0.6, 0.2, 0.2], 
                                    random_seed=42,
                                    random_split=True)
    us_train_traj_set, us_test_traj_set, us_val_traj_set = set_split
    us_train_idx_set, us_test_idx_set, us_val_idx_set = idx_split
    us_train_traj_set = create_action_dataset_v5(dataset=us_train_traj_set, 
                            idx_set=us_train_idx_set,
                            context_len=121,
                                        )
    us_val_traj_set = create_action_dataset_v5(dataset=us_val_traj_set, 
                            idx_set=us_val_idx_set,
                            context_len=121,
                                        )
    us_test_traj_set = create_action_dataset_v5(dataset=us_test_traj_set, 
                            idx_set=us_test_idx_set,
                            context_len=121,
                                        )
    
    # src_stats = us_test_traj_set.get_src_stats()
    # test_idx_set = None #TODO: clean unneeded vars and args

    # read cfg not working and requires postprocessing
    # cfg_path =  tmp_path[:-3] + ".yml"
    # cfg =  read_cfg_file(cfg_path)

    cfg = jugaad_cfg(context_len=121, device='cuda')
    op_traj_dict_list, results = translate(transformer,us_train_idx_set, us_train_traj_set, 
                                            None, cfg)
    os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"),exist_ok=True)
    os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"), exist_ok=True)
    save_object(op_traj_dict_list, os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"))
    save_object(results,os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"))
    

    op_traj_dict_list = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"))
    results = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"))
    _, dummy_target, _, _, dummy_vel_fld_seq, _,_,dummy_flow_dir,_ = us_train_traj_set[0]
    # intantiate gym env for vizualization purposes
    env_4_viz = setup_env(dummy_flow_dir)

    test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
    path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
    all_att_mat_list =  [d['all_att_mat'] for d in op_traj_dict_list]
    success_list = [d['success'] for d in op_traj_dict_list]
    actions = [d['actions'] for d in op_traj_dict_list]
    
    # taken from vis_traj_with_attention.py in decision transformer project
    print(f"model_name = {model_name}")
    save_dir = "paper_plots/"  + model_name + "/" + dataset_name
    # save_dir = "paper_plots/"  + model_name 
    save_dir = join(ROOT,save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    paper_plot_info = {"trajs_by_arr": {"fname":"T_arr"},
                    "trajs_by_att": {"ts":[17,46, 70],"fname":"att"},
                    "att_heatmap":{"fname":"heatmap"},
                    "plot_val_ip_op":{"fname":"val_ip_op"},
                    "plot_actions":{"fname":"test_actions_ip_op_2pi"},
                    "plot_train_val_ip_op":{"fname":"plot_train_val_ip_op"},
                    "loss_avg_returns":{"fname":"loss"},
                    "vel_field":{"fname":"vel_field", "ts":[1,60,119]}
                        }
    pp = paper_plots(env_4_viz, op_traj_dict_list, src_stats,
                        paper_plot_info=paper_plot_info,
                        save_dir=save_dir)
    # pp.plot_val_ip_op(us_train_traj_set, test_set_txy_preds, path_lens, success_list)
    
    pp.plot_actions(us_train_traj_set, test_set_txy_preds, path_lens, success_list, actions)
    
    # pp.plot_traj_by_arr(us_test_traj_set,set_str="_us_test_")
    # pp.plot_train_val_ip_op(train_traj_dataset, val_traj_dataset)
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
    
    plot_all_attention_mats(all_att_mat_list[0],
                            log_wandb=False, 
                            model_name=model_name+"_on_"+dataset_name)
    for t in [i*10 for i in range(1,11)]:
        viz_op_traj_with_attention(test_set_txy_preds,
                            all_att_mat_list, 
                            path_lens,
                            mode='dec_sa',       #or 'a_s_attention'
                            average_across_layers=True,
                            stats=None, 
                            env=env_4_viz, 
                            log_wandb=False, 
                            scale_each_row=True,
                            plot_policy=False,
                            traj_idx=None,      #None=all, list of rzn_ids []
                            show_scatter=False,
                            plot_flow=True,
                            at_time=t,
                            model_name=model_name+"_on_"+dataset_name
                            )
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
    print(f" Results on unseen test")
    print_dict(results)  
    # visualize_input(us_test_traj_set, at_time=119, 
    #                                   env=env_4_viz,
    #                                   log_wandb=False,
    #                                   data_name=dataset_name
    #                                         )

    print(f" Results on unseen test")
    print_dict(results)
    return  


    # Evaluation by translation   
    if epoch % eval_inerval == 0 and rank == 0:
        # print("plotting attention")
        # plot_all_attention_mats(tr_all_att_mat)
        # plot_all_attention_mats(val_all_att_mat)
        print("translating")
        tr_op_traj_dict_list, tr_results = translate(transformer, train_idx_set, tr_set, None, 
                                               cfg, earlybreak=tt_eb[0])
        
        tr_set_txy_preds = [d['states'] for d in tr_op_traj_dict_list]
        all_att_mat_list =  [d['all_att_mat'] for d in tr_op_traj_dict_list]

        # tr_set_txy_PREDS_ = [d['states_for_action_labels'] for d in tr_op_traj_dict_list]

        path_lens = [d['n_tsteps'] for d in tr_op_traj_dict_list]
        visualize_output(tr_set_txy_preds, 
                            path_lens,
                            iter_i = 0, 
                            stats=None, 
                            env=env_4_viz, 
                            log_wandb=True, 
                            plot_policy=False,
                            traj_idx=None,      #None = all, list of rzn_ids []
                            show_scatter=False,
                            at_time=None,
                            color_by_time=True, #TODO: fix tdone issue in src_utils
                            plot_flow=True,
                            wandb_suffix="train")
        
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
        
    
        viz_op_traj_with_attention(tr_set_txy_preds,
                            all_att_mat_list, # could be enc_sa, dec_sa, dec_ga
                            path_lens,
                            mode='dec_sa',       #or 'a_s_attention'
                            average_across_layers=True,
                            stats=None, 
                            env=env_4_viz, 
                            log_wandb=True, 
                            scale_each_row=True,
                            plot_policy=False,
                            traj_idx=None,      #None=all, list of rzn_ids []
                            show_scatter=False,
                            plot_flow=True,
                            at_time=88,
                            model_name="GenHW"+"_on_"+dataset_name
                            )  
                    
        val_op_traj_dict_list, val_results = translate(transformer, val_idx_set, tr_set, None, 
                                                       cfg, earlybreak=tt_eb[1])
        val_set_txy_preds = [d['states'] for d in val_op_traj_dict_list]
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
                            at_time=None,
                            color_by_time=True, #TODO: fix tdone issue in src_utils
                            plot_flow=True,
                            wandb_suffix="val") 

    #Note: val_results get updated after eval_inerval
    translate_avg_ep_len = val_results['translate/avg_ep_len']
    translate_avg_val_loss = val_results['avg_val_loss']
    success_ratio = val_results['translate/success_ratio']
    tr_success_ratio = tr_results['translate/success_ratio']
    count = val_results['runs_from_set(count)'] 
    log_dict = { "tr_loss_vs_epoch (unpadded elems)": train_loss,
        "val_loss_vs_epoch (unpadded elems)": val_loss,
        "avg_val_loss (across pred len)": translate_avg_val_loss,
        "success_ratio": success_ratio,
        "succes_ratio (train)": tr_success_ratio,
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
    if success_ratio > max_sr and rank == 0:
        # min_ETA = translate_avg_ep_len
        max_sr = success_ratio
        print("saving current model at: " + save_model_path)

        best_avg_episode_length = translate_avg_ep_len
        best_success_ratio = success_ratio
        best_epoch = epoch
        # "avg_val_loss"= eval_avg_val_loss

        torch.save(transformer, save_model_path)
        tmp_path = save_model_path[:-1]
        torch.save(transformer, tmp_path)


cfg_copy_path = save_model_path[:-2] + "yml"
save_yaml(cfg_copy_path,cfg)
print(f"cfg_copy_path = {cfg_copy_path}")
wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
wandb.run.summary["best_success_ratio"] = best_success_ratio
wandb.run.summary["total_runs_val_set"] = len(val_op_traj_dict_list)
wandb.run.summary["best_epoch"] = best_epoch

print("=" * 60)
print("finished training!")
print("=" * 60)

print(f"\n\n ---- running inference on test set ----- \n\n")
transformer = torch.load(tmp_path)
op_traj_dict_list, results  = translate(transformer,test_idx_set, test_set, 
                                        None, cfg, earlybreak=tt_eb[2])
test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
path_lens = [d['n_tsteps'] for d in op_traj_dict_list]  
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
                    wandb_suffix="test")

end_time = datetime.now().replace(microsecond=0)
end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
print("started training at: " + start_time_str)
print("finished training at: " + end_time_str)
print("total training time: " + time_elapsed)
print("saved last updated model at: " + save_model_path)
print("=" * 60)





# def load_prev_and_test(args, cfg_name):

#     # load model
#     # tmp_path = ROOT + "log/my_translat_GPTdset_DG3_model_04-01-03-20.pt"
#     # ROOT = /home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/
#     tmp_path = ROOT + "log/my_translat_DOLS_Cylinder_model_06-21-14-57.pt" 
#     transformer = torch.load(tmp_path)
#     model_name = tmp_path[:-3].split('/')[-1]
#     # load unseen dataset
#     targ = '5'
#     dset = 'test'
#     dataset_path = ROOT + f"data/DOLS_Cylinder/targ_{targ}/gathered_targ_{targ}.pkl"
#     traj_dataset = load_pkl(dataset_path)
#     dataset_name = dataset_path[:-4].split('/')[-1]
#     # src_stats_path = tmp_path[:-3] + "_src_stats.npy"
#     src_stats_path = ROOT + f"log/{model_name}_src_stats.npy"
#     src_stats = np.load(src_stats_path)
#     src_stats = (src_stats[0], src_stats[1])

#     idx_split, set_split = get_data_split(traj_dataset,
#                                     split_ratio=[0.8,0.05,0.15], 
#                                     random_seed=42,
#                                     random_split=True)
#     us_train_traj_set, us_test_traj_set, us_val_traj_set = set_split
#     us_train_idx_set, us_test_idx_set, us_val_idx_set = idx_split
#     us_train_traj_set = create_action_dataset_v2(us_train_traj_set, 
#                             idx_set=[None],
#                             context_len=101,
#                             # norm_params_4_val = src_stats
#                                         )
#     # _, _, us_test_traj_set = set_split
#     # _, _, us_test_idx_set = idx_split
#     us_val_traj_set = create_action_dataset_v2(us_val_traj_set, 
#                             idx_set=[None],
#                             context_len=101,
#                             norm_params_4_val = src_stats
#                                         )
#     us_test_traj_set = create_action_dataset_v2(us_test_traj_set, 
#                             idx_set=[None],
#                             context_len=101,
#                             norm_params_4_val = src_stats
#                                         )
    
#     # src_stats = us_test_traj_set.get_src_stats()
#     test_idx_set = None #TODO: clean unneeded vars and args
#     # read cfg not working and requires postprocessing 
#     # cfg_path =  tmp_path[:-3] + ".yml"
#     # cfg =  read_cfg_file(cfg_path)
#     cfg = jugaad_cfg(context_len=101, device='cuda')
#     # translate_start_time = timer()


#     op_traj_dict_list, results = translate(transformer,us_test_idx_set, us_test_traj_set, 
#                                             None, cfg, earlybreak=500)
#     # translate_end_time = timer()
#     # print(f"Translate runtime = {(translate_end_time - translate_start_time):.3f}s")
#     os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/DOLS_targ_{targ}/{dset}_op_traj_dict_list.pkl"),exist_ok=True)
#     os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/DOLS_targ_{targ}/{dset}_results.pkl"), exist_ok=True)
#     save_object(op_traj_dict_list, os.path.join(ROOT, f"paper_plots/{model_name}/DOLS_targ_{targ}/{dset}_op_traj_dict_list.pkl"))
#     save_object(results,os.path.join(ROOT, f"paper_plots/{model_name}/DOLS_targ_{targ}/{dset}_results.pkl"))

#     op_traj_dict_list = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/DOLS_targ_{targ}/{dset}_op_traj_dict_list.pkl"))
#     results = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/DOLS_targ_{targ}/{dset}_results.pkl"))
#     _, dummy_target, _, _, dummy_env_coef_seq, _,_,dummy_flow_dir,_ = us_val_traj_set[0]
#     # intantiate gym env for vizualization purposes
#     env_4_viz = setup_env(dummy_flow_dir)

#     test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
#     path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
#     all_att_mat_list =  [d['all_att_mat'] for d in op_traj_dict_list]
#     success_list = [d['success'] for d in op_traj_dict_list]
#     actions = [d['actions'] for d in op_traj_dict_list]
    
#     # taken from vis_traj_with_attention.py in decision transformer project
#     print(f"model_name = {model_name}")
#     save_dir = "paper_plots/"  + model_name + f"/DOLS_targ_{targ}/increased_cbar"
#     save_dir = join(ROOT,save_dir)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     paper_plot_info = {"trajs_by_arr": {"fname":"T_arr"},
#                     "trajs_by_att": {"ts":[17,46, 70],"fname":"att"},
#                     "att_heatmap":{"fname":"heatmap"},
#                     "plot_val_ip_op":{"fname":"test_ip_op"},
#                     "plot_trajs_ip_op":{"fname":"test_trajs_ip_op"},
#                     "plot_actions":{"fname":"test_actions_ip_op"},
#                     "plot_train_val_ip_op":{"fname":"plot_train_val_ip_op"},
#                     "loss_avg_returns":{"fname":"loss"},
#                     "vel_field":{"fname":"vel_field", "ts":[1,60,119]}
#                         }
#     pp = paper_plots(env_4_viz, op_traj_dict_list, src_stats,
#                         paper_plot_info=paper_plot_info,
#                         save_dir=save_dir)
#     pp.plot_val_ip_op(us_test_traj_set, test_set_txy_preds, path_lens, success_list)
#     pp.plot_trajs_ip_op(us_test_traj_set, test_set_txy_preds, path_lens, success_list)