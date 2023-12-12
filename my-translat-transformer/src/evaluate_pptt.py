import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from timeit import default_timer as timer

from src_utils import create_envEnc_dtDec_dataset, compare_trajectories, viz_op_traj_with_attention
from src_utils import get_data_split, generate_square_subsequent_mask, denormalize, visualize_output, visualize_input
from src_utils import see_steplr_trend, simulate_tgt_actions, plot_attention_weights, checkpoint_model
from src_utils import setup_env
from utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object, make_dir, convert_dict_to_obj
from paper_plots_pptt import paper_plots
import gym
import gym_examples
import sys
import pickle
import wandb
import imageio.v2 as imageio
import argparse
from os.path import join
from datetime import datetime
import numpy as np
from root_path import ROOT
from PIL import ImageFile
sys.path.append("/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer")
from extract_rep_3channel import ExtractRep
import tqdm
import matplotlib.pyplot as plt
tmp_path='/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tmp'

def predict_env_embeddings_autoreg(env_predictor, env_pred_input, timesteps, input_mask, t, cl):
    """
    env_predictor: model
    t: timestep to start prediction from
    cl:  context len
    env_pred_input: input tensor of env embeddings contain observed embeddings till timestep t
    input_mask: mask with padding 

    env_pred_input itself is updated with predictions.
    when returning, env_pred_input[0,0:t+1,:] has observed embeddings
                    env_pred_input[0, t+1:,:] has predicted embeddings
    it can be directly used as the dyn_encoder_input for the pptt encoder
    """ 

    for p in range(t+1, cl-1):
        input_mask[0,p-1] = False
        ag_pred_at_p = env_predictor(timesteps, env_pred_input, padding_mask=input_mask)[0,p-1]
        env_pred_input[0,p,:] = ag_pred_at_p

    return env_pred_input

def closed_loop_tranlsate(model: torch.nn.Module, test_idx, test_dataloader, tr_set_stats, cfg, autoenc, env_predictor, earlybreak=10**8):
    model.eval()
    count = 0           # keeps count of total episodes
    success_count = 0   # keeps count of successful episodes
    op_traj_dict_list = []
    total_reward = 0
    states_mean, states_std = tr_set_stats
    states_mean = torch.from_numpy(states_mean).to(cfg.device)
    states_std = torch.from_numpy(states_std).to(cfg.device)
    eps = 1e-6
    cl = cfg.context_len
    device = cfg.device
    # extractor = ExtractRep(None, autoenc)

    with torch.no_grad():
        for _, (timesteps, actions, states, rtg, action_mask, final_pos, encoder_input_labels, tgt_padding_mask, n, idx, flow_dir, rzn) in enumerate(tqdm.tqdm(test_dataloader)):
            # if idx%100==0:
            #     print(idx)
            states = states.to(device)
            obs_embs = encoder_input_labels.to(device) # these are observed at each timestep
            final_pos = final_pos.to(device)
            timesteps = timesteps.to(device)
            op_traj_dict = {}
            reached_target = False

            # set up environment
            flow_dir = flow_dir[0]
            env = setup_env(flow_dir)
            env.reset()
            env.set_rzn(rzn)

            count += 1
            if count == earlybreak:
                break

            # TODO: verify whether lenght should be cl 
            pred_actions = torch.zeros_like(actions, device=device)
            pred_states = torch.zeros_like(states, device=device)
            pred_rtg = torch.zeros_like(rtg, device=device)
            tgt_padding_mask = torch.ones_like(tgt_padding_mask).type(torch.bool).to(device)
            env_predictor_padding_mask = torch.ones_like(timesteps).type(torch.bool).to(device)
            # TODO: verify correctness of sequence length
            dyn_encoder_input = torch.zeros_like(obs_embs, device=device) # [1, 123, 768]
            env_predictor_input = torch.zeros_like(obs_embs)

            # TODO: start state can taken from cfg as well. Current is ok
            # since not trying to generalize over start states
            pred_states[0,0,:] = (states[0,0,:] - states_mean) / (states_std + eps)
            # TODO: might have to not condition on desired rtg if doesnt work well
            running_rtg = cfg.desired_rtg  / cfg.rtg_scale 
            pred_rtg[0,0,0] = running_rtg  # scaled
            episode_returns = 0
            running_state = np.array(env.reset())


            for t in range(0, cl):
                if t < cl:
                    # t=0: Xf R s           0:3
                    # t=1: Xf R s a R s     0:6
                    tgt_padding_mask[0, :3*(t+1)] = False
                    #Reinitialise mask to all Trues
                    env_predictor_padding_mask[0,:] = True
                    env_predictor_padding_mask[0, :t+1] = False #shape [1,124]
                    # TODO: can put in src_padding_mask for surity
                    # return form src_util loader
                    
                    # get observed env embedding till time t and feed it to the tdec as input
                    env_predictor_input[0,:t+1,:] = obs_embs[0,:t+1,:] #shape [1,123,1024]
                    # -1 because timesteps in env_predictor start from 0
                    # :cl because timesteps here range from 0:cl+1
                    # IMP: train env_predictor with the same context len
                    #  shape [1,123,1024]
                    # env_predictor_pred = env_predictor(timesteps[:,:cl]-1, env_predictor_input[:,:cl], env_predictor_padding_mask[:,:cl])
                    env_predictor_pred = predict_env_embeddings_autoreg(env_predictor,
                                                                            env_predictor_input[:,:cl],         #shape [1,123,1024]
                                                                            timesteps[:,:cl]-1,                 #shape [1,123]
                                                                            env_predictor_padding_mask[:,:cl], #shape [1,123]
                                                                            t,cl)
                    
                    # dyn_encoder_input[0,:t+1,:] = env_predictor_input[0,:t+1,:] 
                    # dyn_encoder_input[0,t+1:,:] = env_predictor_pred[0,t+1:,:]
                    # env_predictor_pred has both observed embeddings and predicted embeddings
                    dyn_encoder_input[0,:,:] = env_predictor_pred[0,:,:]
                    # plt.ylim([0.4,0.8])
                    # plt.plot(dyn_encoder_input[0,:t+1,0].cpu().numpy(),'--o')
                    # plt.plot(dyn_encoder_input[0,:20,0].cpu().numpy(),'-*')
                    # plt.savefig(join(tmp_path,'test_dyn_enc_ip.png'))

                    _pred_actions_ = model.forward( src=dyn_encoder_input,
                                                    tgt_states=pred_states,
                                                    tgt_actions=pred_actions,
                                                    tgt_rtg=pred_rtg,
                                                    tgt_final_pos=final_pos,
                                                    src_mask=None,
                                                    tgt_mask=generate_square_subsequent_mask(model.max_dec_len, device),
                                                    src_padding_mask=None,
                                                    tgt_padding_mask=tgt_padding_mask,
                                                    memory_key_padding_mask=None,
                                                    timesteps=timesteps,
                                                    )
                 
                    act = _pred_actions_[0, t].detach()
                else:
                    # TODO: complete later or remove if-else condition
                    raise ValueError("case should be impossible")
     
                running_state, running_reward, done, info = env.step(act.cpu().numpy()*2*np.pi)
                running_state = torch.from_numpy(running_state).to(device)
                # add action in placeholder
                pred_actions[0, t] = act
                pred_states[0, t+1] = (running_state - states_mean) / (states_std + eps)
                running_rtg = running_rtg - (running_reward / cfg.rtg_scale)
                pred_rtg[0,t+1,0] = running_rtg   # scaled
               
                total_reward += running_reward
                episode_returns += running_reward

                if done:
                    t_done = t + 1
                    # if done and getting positive reward
                    if running_reward > 0: 
                        reached_target = True
                        success_count += 1
                    break
                # end_fp = time.time()
                # print(f"rebuttal: forward pass time: {end_fp - start_fp}")

            op_traj_dict['states'] = states.cpu() # normalized
            op_traj_dict['actions'] = actions.cpu()*2*np.pi
            op_traj_dict['t_done'] = t_done
            op_traj_dict['n_tsteps'] = t+2
            op_traj_dict['success'] = reached_target
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
    results['translate/success_ratio'] = success_count / count
    results['runs_from_set(count)'] = count
    return op_traj_dict_list, results
    


def translate(model: torch.nn.Module, test_idx, test_dataloader, tr_set_stats, cfg, earlybreak=10**8):
    model.eval()
    count = 0           # keeps count of total episodes
    success_count = 0   # keeps count of successful episodes
    op_traj_dict_list = []
    total_timesteps = 0  # TODO: comment description here
    total_reward = 0
    states_mean, states_std = tr_set_stats
    states_mean = torch.from_numpy(states_mean).to(cfg.device)
    states_std = torch.from_numpy(states_std).to(cfg.device)
    eps = 1e-6
    cl = cfg.context_len
    device = cfg.device
    
    with torch.no_grad():
        # TODO: make closed loop inference
        for timesteps, actions, states, rtg, action_mask, final_pos, encoder_input, tgt_padding_mask, n, idx, flow_dir, rzn in test_dataloader:
            # if idx%100==0:
            #     print(idx)
            states = states.to(device)
            # TODO: will have to use dynamically updated encoder_input instead
            encoder_input = encoder_input.to(device)
            final_pos = final_pos.to(device)
            timesteps = timesteps.to(device)
            op_traj_dict = {}
            reached_target = False

            # set up environment
            flow_dir = flow_dir[0]
            env = setup_env(flow_dir)
            env.reset()
            env.set_rzn(rzn)

            count += 1
            if count == earlybreak:
                break

            # TODO: verify whether lenght should be cl 
            pred_actions = torch.zeros_like(actions, device=device)
            pred_states = torch.zeros_like(states, device=device)
            pred_rtg = torch.zeros_like(rtg, device=device)
            tgt_padding_mask = torch.ones_like(tgt_padding_mask).type(torch.bool).to(device)
            
            # TODO: start state can taken from cfg as well. Current is ok
            # since not trying to generalize over start states
            #
            # states is arleady normalized in the dataloader
            pred_states[0,0,:] = states[0,0,:]
            # TODO: might have to not condition on desired rtg if doesnt work well
            running_rtg = cfg.desired_rtg  / cfg.rtg_scale 
            pred_rtg[0,0,0] = running_rtg  # scaled
            episode_returns = 0
            env.reset()
            # for debuggimg
            dss = torch.from_numpy(np.array(env.reset())).to(device)
            dss = (dss - states_mean) / (states_std + eps)
            assert(torch.all(dss == pred_states[0,0]).item()), "state[0,0] !=  env.reset() "
            # TODO: IMPPP! is running state in the above line same as
            # states[0,0,:] # SHUBHAM

            for t in range(0, cl):
                total_timesteps += 1
                if t < cl:
                    # t=0: Xf R s           0:3
                    # t=1: Xf R s a R s     0:6
                    tgt_padding_mask[0, :3*(t+1)] = False
                    # TODO: can put in src_padding_mask for surity
                    # return form src_util loader
                    _pred_actions_ = model.forward( src=encoder_input,
                                                    tgt_states=pred_states,
                                                    tgt_actions=pred_actions,
                                                    tgt_rtg=pred_rtg,
                                                    tgt_final_pos=final_pos,
                                                    src_mask=None,
                                                    tgt_mask=generate_square_subsequent_mask(model.max_dec_len, device),
                                                    src_padding_mask=None,
                                                    tgt_padding_mask=tgt_padding_mask,
                                                    memory_key_padding_mask=None,
                                                    timesteps=timesteps,
                                                    )
                 
                    act = _pred_actions_[0, t].detach()
                else:
                    # TODO: complete later or remove if-else condition
                    raise ValueError("case should be impossible")
     
                running_state, running_reward, done, info = env.step(act.cpu().numpy()*2*np.pi)
                running_state = torch.from_numpy(running_state).to(device)
                # add action in placeholder
                pred_actions[0, t] = act
                pred_states[0, t+1] = (running_state - states_mean) / (states_std + eps)
                running_rtg = running_rtg - (running_reward / cfg.rtg_scale)
                pred_rtg[0,t+1,0] = running_rtg   # scaled
               
                total_reward += running_reward
                episode_returns += running_reward

                if done:
                    t_done = t + 1
                    # if done and getting positive reward
                    if running_reward > 0: 
                        reached_target = True
                        success_count += 1
                    break
                # end_fp = time.time()
                # print(f"rebuttal: forward pass time: {end_fp - start_fp}")

            op_traj_dict['states'] = states.cpu() # normalized
            op_traj_dict['actions'] = actions.cpu()*2*np.pi
            op_traj_dict['t_done'] = t_done
            op_traj_dict['n_tsteps'] = t+2
            op_traj_dict['success'] = reached_target
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
    results['translate/success_ratio'] = success_count / count
    results['runs_from_set(count)'] = count
    return op_traj_dict_list, results

           # print("plotting attention")
            # plot_all_attention_mats(tr_all_att_mat)
            # plot_all_attention_mats(val_all_att_mat)
            # print("translating")
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
         
    #         val_op_traj_dict_list, val_results = translate(transformer, None, test_dataloader, states_stats, 
    #                                                        cfg, earlybreak=tt_eb[2])
    #         val_set_txy_preds = [d['states']*states_stats[1] + states_stats[0] for d in val_op_traj_dict_list]
    #         path_lens = [d['n_tsteps'] for d in val_op_traj_dict_list]
    #         visualize_output(val_set_txy_preds, 
    #                             path_lens,
    #                             iter_i = 0, 
    #                             stats=None, 
    #                             env=env_4_viz, 
    #                             log_wandb=True, 
    #                             plot_policy=False,
    #                             traj_idx=None,      #None=all, list of rzn_ids []
    #                             show_scatter=True,
    #                             at_time=None,
    #                             color_by_time=True, #TODO: fix tdone issue in src_utils
    #                             plot_flow=True,
    #                             wandb_suffix="val") 


    #     translate_avg_ep_len = val_results['translate/avg_ep_len']
    #     # translate_avg_val_loss = val_results['avg_val_loss']
    #     success_ratio = val_results['translate/success_ratio']
    #     # tr_success_ratio = tr_results['translate/success_ratio']
    #     count = val_results['runs_from_set(count)'] 
        
    #     log_dict = { "tr_loss_vs_epoch (unpadded elems)": train_loss,
    #         "val_loss_vs_epoch (unpadded elems)": val_loss,
    #         # "avg_val_loss (across pred len)": translate_avg_val_loss,
    #         "success_ratio": success_ratio,
    #         # "succes_ratio (train)": tr_success_ratio,
    #         'runs_from_set(count)': count,
    #         "ETA": translate_avg_ep_len,
    #         # "lr" : scheduler.get_last_lr()[0] if use_scheduler else lr
    #         }
    #     wandb.log(log_dict)
        
        
    # wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    # wandb.run.summary["best_success_ratio"] = best_success_ratio
    # wandb.run.summary["total_runs_val_set"] = len(val_op_traj_dict_list)
    # wandb.run.summary["best_epoch"] = best_epoch
    
    
    #     print(f"\n\n ---- running inference on test set ----- \n\n")
    # load_path = join(model_save_dir, f"")
    # transformer = torch.load(tmp_path)
    # op_traj_dict_list, results  = translate(transformer,None, test_dataloader, 
    #                                         states_stats, cfg, earlybreak=tt_eb[1])
    # test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
    # path_lens = [d['n_tsteps'] for d in op_traj_dict_list]  
    # visualize_output(test_set_txy_preds, 
    #                     path_lens,
    #                     iter_i = 0, 
    #                     stats=None, 
    #                     env=env_4_viz, 
    #                     log_wandb=True, 
    #                     plot_policy=False,
    #                     traj_idx=None,      #None=all, list of rzn_ids []
    #                     show_scatter=True,
    #                     at_time=None,
    #                     color_by_time=True, #TODO: fix tdone issue in src_utils
    #                     plot_flow=True,
    #                     wandb_suffix="test")

"""
TODO:
0. save to same wandb run as used in train (high) DONE
1. Make args for running inference on train, test, val sets with tt_eb from cfg (low)
2. override tt_eb from cfg using cmd line args (low)
4. fix paper_plots for pptt
    - scale issue
    - need obstacle info
    - remove all hardcode numbers and provide a dictionary of plot_params instead (high)
"""

class inference:

    def __init__(self, args):
        self.args = args
        # load run cfg
        cfg_path = join(args.log_exp_dir, "run_cfg.yml")
        cfg = read_cfg_file(cfg_path)
        cfg = convert_dict_to_obj(cfg)
        self.cfg = cfg

        if args.test_data != None:
            # TODO: correct if throws error due to zeros in list
            dataset_path = args.test_data
            self.split_ratio = [0, 1, 0]
        else:
            dataset_path =  join(ROOT,cfg.dataset_path)
            self.split_ratio = cfg.split_tr_tst_val


        states_stats_path = join(args.log_exp_dir, "states_stats.npy")
        states_stats = list(np.load(states_stats_path))

        dataset_name = dataset_path.split('/')[-1].split('.')[0]
        inference_path = join(args.log_exp_dir, 'inference') # path to subfolder named inference
        make_dir(inference_path)

        dataset_subdir = join(inference_path, dataset_name)
        make_dir(dataset_subdir) 

        results_path = join(dataset_subdir, args.model_state[:-3]) # remove .pt
        make_dir(results_path, exist_ok=True) # overwrites folder 

        device = torch.device(cfg.device)

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
        
        
        env_predictor = torch.load(f"/{join(*cfg.env_predictor_name.split('/')[:-1])}/arch.pt")
        env_predictor.load_state_dict(torch.load(cfg.env_predictor_name)['model_state_dict'])
        
        idx_split, set_split = get_data_split(traj_dataset,
                                            split_ratio=self.split_ratio, 
                                            random_seed=cfg.split_ran_seed, 
                                            random_split=cfg.random_split)

        us_train_traj_set, us_test_traj_set, us_val_traj_set = set_split
        us_train_idx_set, us_test_idx_set, us_val_idx_set = idx_split

        us_test_traj_set = create_envEnc_dtDec_dataset(us_test_traj_set, 
                                [None],
                                cfg.context_len,
                                autoenc,
                                cfg.rtg_scale,
                                norm_params_4_val = states_stats,
                                ae_type=ae_type,
                                device=device)
        self.test_dataloader = DataLoader(us_test_traj_set, batch_size=1, shuffle=True)

        self.dataset_path = dataset_path
        self.inference_path = inference_path
        self.dataset_subdir = dataset_subdir
        self.results_path = results_path
        self.states_stats_path = states_stats_path
        self.states_stats =states_stats
        self.us_test_traj_set = us_test_traj_set
        self.dset = 'test' # hard coded. TODO: 
        self.autoenc = autoenc
        self.env_predictor = env_predictor



    def load_and_translate(self,closed_loop_translation=True):

        args = self.args
        cfg = self.cfg

        # Load model architecture
        model_path =  join(args.log_exp_dir, "arch.pt")
        transformer = torch.load(model_path)

        # Load model weights
        model_state_dict_path = join(args.log_exp_dir, args.model_state)
        transformer.load_state_dict(torch.load(model_state_dict_path)['model_state_dict'])

        # TODO: save to same wandb run as used in train (Shubham)
        wandb.init(project = "envEnc_dtDec",id = cfg.wb_id, resume=True)
        wandb.log({'test': 100})

        print(f"\n----- {closed_loop_translation=} ------\n")
        if closed_loop_translation:
            op_traj_dict_list, results = closed_loop_tranlsate(transformer,None, self.test_dataloader, 
                                                self.states_stats, cfg, 
                                                autoenc=self.autoenc,
                                                env_predictor=self.env_predictor,
                                                earlybreak=10)
        else:
            op_traj_dict_list, results = translate(transformer,None, self.test_dataloader, 
                                                    self.states_stats, cfg, earlybreak=10)
                   
        # TODO: remove hardcode
        # translate_end_time = timer()
        # print(f"Translate runtime = {(translate_end_time - translate_start_time):.3f}s")

        save_object(op_traj_dict_list, join(self.results_path, f"{self.dset}_op_traj_dict_list.pkl"))
        save_object(results, join(self.results_path, f"{self.dset}_results.pkl"))

        return


    def make_paper_plots(self):

        # # TODO: load for viz only
        op_traj_dict_list = load_pkl(join(self.results_path, f"{self.dset}_op_traj_dict_list.pkl"))
        # results = load_pkl( join(self.results_path, f"{self.dset}_results.pkl"))

        dummy_flow_dir = self.us_test_traj_set[0][-2]
        # intantiate gym env for vizualization purposes
        env_4_viz = setup_env(dummy_flow_dir)
        obs_mask = np.load(join(dummy_flow_dir, "obstacle_mask.npy"))

        states_mean, states_std = self.states_stats
        test_set_txy_preds = [d['states']*states_std + states_mean for d in op_traj_dict_list]
        path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
        # all_att_mat_list =  [d['all_att_mat'] for d in op_traj_dict_list]
        success_list = [d['success'] for d in op_traj_dict_list]
        actions = [d['actions'] for d in op_traj_dict_list]

        # taken from vis_traj_with_attention.py in decision transformer project

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

        pp = paper_plots(env_4_viz, op_traj_dict_list, self.states_stats,
                            paper_plot_info=paper_plot_info,
                            # save_dir=join(self.results_path, "temp_test")
                            save_dir=self.results_path 
                            )
        
        pp.plot_val_ip_op(self.us_test_traj_set, obs_mask[1,:], test_set_txy_preds, path_lens, success_list)

        return

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


if __name__ == "__main__":

    # log/exp_dir cotains model, train config, saved_states
    # plots from translate will be stored in the same dir
    # def_log_exp_dir = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/log/my_EnvE_DtD_GPT_DG3_model_10-30-15-12-47"
    def_log_exp_dir = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/log/my_EnvE_DtD_GPT_DG3_model_12-04-17-05-45"
    
    print(f"cuda available: {torch.cuda.is_available()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_exp_dir', type=str, default=def_log_exp_dir)
    # enter state (eg. ep20.pt) from cmd line
    parser.add_argument('--model_state', type=str, default="best_vloss.pt")
    # takes test set of original dataset from which training and validation was done
    # to test on unseen data, put dataset path as --test_data argument in cmd line
    # all of the unseen data will be used as the test set
    parser.add_argument('--test_data', type=str, default=None) 
    # use to re-plot without translating again
    parser.add_argument('--plot_only', type=bool, default=False)
    args = parser.parse_args()

    infer = inference(args)  
    if not args.plot_only:   
        infer.load_and_translate()   
    infer.make_paper_plots()

