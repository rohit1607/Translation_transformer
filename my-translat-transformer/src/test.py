 
 
 
 
 
 
 
  
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