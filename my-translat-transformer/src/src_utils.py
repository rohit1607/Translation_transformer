import torch
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle
import wandb
from sklearn.model_selection import train_test_split
from operator import itemgetter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from os.path import join
import sys
import time
import re
from pathlib import Path
import gym
import gym_examples
import matplotlib
sys.path.append("/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer")
from src.utils import read_cfg_file, execution_time
from src.root_path import ROOT
# from root_path import ROOT
# from extract_rep_3channel import ExtractRep as ExtractRep_mae
from extract_rep_3channel_ae import ExtractRep as ExtractRep_tae
from PIL import Image
import os


@execution_time
def get_next_item(loader):
    return next(loader)

def setup_env(flow_dir, target_id=0, denorm_final_pos=None):
    flow_specific_cfg = read_cfg_file(cfg_name=join(flow_dir,"cfg_used_in_proc_data.yml"))
    env_name = flow_specific_cfg["env_name"]
    params2 = read_cfg_file(cfg_name=join(flow_dir,"params.yml"))
    env = gym.make(env_name)
    # # IMP: CLEAN CODE
    # env.if_scale_velocity = True
    env.setup(flow_specific_cfg, params2, add_trans_noise=False, target_id=target_id, denorm_final_pos=denorm_final_pos) # dummy target id, will be overwritten later
    return env


def checkpoint_model(model, save_dir, epoch, optimizer, 
                     scheduler=None, 
                     loss=None, 
                     save_type ='cur_best', # or "some_epoch"
                     only_save_states=True):
    """
    save_type: cur_best overwrites the current best model based on val_loss
                some_epoch save states at given epoch
    only_save_states: saves only state_dicts if true. saves full model as well if False
    loss: (avg_tr_loss, avg_val_loss) at epoch 
    """
    # save full model. it will have architecture info as well
    if not only_save_states:
        save_path = join(save_dir, "arch.pt")
        torch.save(model, save_path)
        return
    
    # save states
    if save_type == 'cur_best':
        save_path = join(save_dir, f"best_vloss.pt")
    elif save_type == "some_epoch":
        save_path = join(save_dir, f"ep{epoch}.pt")
    elif save_type == 'cur_best_translation':
        save_path = join(save_dir, f"best_translation_loss.pt")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'tr_loss': loss[0],
            'val_loss': loss[1],
            }, save_path)
    
    return


def get_data_split(traj_dataset, split_ratio=[0.6, 0.2, 0.2], random_seed=42, random_split=True):
    """"

    traj_dataset: contains traj infor for all rzns
                    list of dictionaries. dictionary contains states, actions, rtgs
    returns:
    idx_split: test_idx, train_idx, val_idx
    set_split:
    """
    n_trajs = len(traj_dataset)
    all_idx = [i for i in range(n_trajs)]
    # split all idx to train and (test+val)
    # test_val_size = split_ratio[1] + split_ratio[2]
    test_val_size = round((split_ratio[1] + split_ratio[2]),1)
    if random_split:
        train_idx, test_val_idx = train_test_split(all_idx, 
                                            test_size=test_val_size,
                                            random_state=random_seed,
                                            shuffle=True)
        # split (test+val into test and val)
        val_size = split_ratio[2]/test_val_size
        test_idx, val_idx = train_test_split(test_val_idx, test_size=val_size, random_state=random_seed)
        idx_split = (train_idx, test_idx, val_idx)
    else:
        train_idx, test_val_idx = train_test_split(all_idx, 
                                    test_size=test_val_size,
                                    shuffle=False)
        # split (test+val into test and val)
        val_size = split_ratio[2]/test_val_size
        test_idx, val_idx = train_test_split(test_val_idx, test_size=val_size, shuffle=False)
        idx_split = (train_idx, test_idx, val_idx)

    train_traj_set = itemgetter(*train_idx)(traj_dataset)
    test_traj_set = itemgetter(*test_idx)(traj_dataset)
    val_traj_set = itemgetter(*val_idx)(traj_dataset)
    
    set_split = (train_traj_set, test_traj_set, val_traj_set)

    return idx_split, set_split


class create_waypoint_dataset(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        env,        #env object
                        context_len, 
                        norm_params_4_val=None):
        """
        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """


        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = [traj for traj in dataset if traj['done']]
        print(f"\n Making dataset out of successful trajectories. \n \
                No. of successful trajs / total trajs = {len(self.dataset)} / {self.n_trajs } \n")
        self.Yi = env.Yi    #Yi are env coeffs
        
        # store waypoints across realizations
        states = [] 
        for traj in self.dataset:
            states.append(traj['states'])
        print(f" states[0].shape:  {states[0].shape}")
        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        # normalize states
        if norm_params_4_val == None:
            for traj in self.dataset:
                traj['states'] = (traj['states'] - self.state_mean) / self.state_std        
        else:
            tr_state_mean,  tr_state_std =  norm_params_4_val
            for traj in self.dataset:
                traj['states'] = (traj['states'] - tr_state_mean) / tr_state_std

    def get_state_stats(self):
        return (self.state_mean, self.state_std)


    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        traj = self.dataset[idx]
        traj_len = traj['states'].shape[0]
        env_coef_seq = self.Yi[:self.context_len, idx, :] # Yi.shape = (nT, nrzns, nmodes)
        # print(f"****** VERIFY: traj_len = {traj_len}")
        padding_len = None
        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # NOTE: add extra padde
            states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            # actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            target_state = torch.from_numpy(traj['target_pos'])
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            states = torch.from_numpy(traj['states'])
            # print(f"+++ in cwg: states.shape = {states.shape}")

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            states = torch.cat([states,
                                torch.zeros(([padding_len+1] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
            target_state = torch.from_numpy(traj['target_pos'])

        dummy_t = float(int(0.8*self.context_len)) # time stamp for target state.
        target_state = np.insert(target_state,0,dummy_t,axis=0)
        # print(f"### verify: target_state = {target_state}")
        return  timesteps, states, traj_mask, target_state, env_coef_seq, traj_len


class create_action_dataset(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        env,        #env object
                        context_len, 
                        norm_params_4_val=None):
        """
        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """

        self.env = env
        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = [traj for traj in dataset if traj['done']]
        print(f"\n Making dataset out of successful trajectories. \n \
                No. of successful trajs / total trajs = {len(self.dataset)} / {self.n_trajs } \n")
        self.Yi = env.Yi    #Yi are env coeffs
        
        # store actions across realizations
        # TODO: try action normalization

        # actions = []
        # for traj in self.dataset:
        #     actions.append(torch.from_numpy(traj['actions']))
        # print(f" actions[0].shape:  {actions[0].shape}")

        # states = [] 
        # for traj in self.dataset:
        #     states.append(traj['states'])
        # print(f" states[0].shape:  {states[0].shape}")
        # # used for input normalization
        # states = np.concatenate(states, axis=0)
        # self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        # # normalize states
        # if norm_params_4_val == None:
        #     for traj in self.dataset:
        #         traj['states'] = (traj['states'] - self.state_mean) / self.state_std        
        # else:
        #     tr_state_mean,  tr_state_std =  norm_params_4_val
        #     for traj in self.dataset:
        #         traj['states'] = (traj['states'] - tr_state_mean) / tr_state_std

    def get_state_stats(self):
        return (self.state_mean, self.state_std)


    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        traj = self.dataset[idx]
        traj_len = traj['states'].shape[0]
        env_coef_seq = self.Yi[:self.context_len, idx, :] # Yi.shape = (nT, nrzns, nmodes)
        # print(f"****** VERIFY: traj_len = {traj_len}")
        padding_len = None
        if traj_len >= self.context_len:
            # TODO: correcly write if condition
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            # states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # # NOTE: add extra padde
            # states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            try:
                actions = torch.from_numpy(traj['actions'][si : si + self.context_len + 1])
            except:
                actions = torch.cat([actions,
                                    torch.zeros(([1] + list(actions.shape[1:])),
                                    dtype=actions.dtype)],
                                    dim=0)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            # target_state = torch.from_numpy(traj['target_pos'])
            print(f" entering if condition - should not happen for basic cases")
            print(f"actions.shape = {actions.shape}")
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(traj['actions'])
            # print(f"+++ in cwg: states.shape = {states.shape}")

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions,
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)
            # print(f" in cwg: states.shape = {states.shape}")

            # sys.exit()
            # actions = torch.from_numpy(traj['actions'])
            # actions = torch.cat([actions,
            #                     torch.zeros(([padding_len] + list(actions.shape[1:])),
            #                     dtype=actions.dtype)],
            #                    dim=0)

            # returns_to_go = torch.from_numpy(traj['returns_to_go'])
            # returns_to_go = torch.cat([returns_to_go,
            #                     torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
            #                     dtype=returns_to_go.dtype)],
            #                    dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        try:
            target_state = torch.from_numpy(traj['target_pos'])
        except:
            target_state = torch.from_numpy(self.env.target_pos)

        # print(f"### verify: target_state = {target_state}")
        # sys.exit()
        dummy_t = float(int(traj_len)) # time stamp for target state.
        if len(target_state.shape) == 2:
            target_state = np.insert(target_state,0,dummy_t,axis=1)
        elif len(target_state.shape) == 1:
            target_state = np.insert(target_state,0,dummy_t,axis=0)

        # print(f"### verify: target_state = {target_state}")
        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx


class create_action_dataset_v2(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        norm_params_4_val=None):
        """
        Different from v1: 
            - deals with a combined dataset made from multiple
              datasets with varying obstacle configs
            - dataset is cleaned.

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """

        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = dataset
        # self.X = np.array([np.concatenate((item[0], item[1]), axis=-1) for item in self.dataset])
        self.X = np.array([item[0] for item in self.dataset])
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)

        # extract actions (tgt) and scale them to range [0,1)
        self.Y = [item[2]/(2*np.pi) for item in self.dataset]

        # normalzise
        if norm_params_4_val == None:
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - self.X_mean
                self.X[i] = np.divide(self.X[i], self.X_std)
                # self.Y[i] = self.Y[i] - self.Y_mean
                # self.Y[i] = np.divide(self.Y[i], self.Y_std)
        else:
            tr_X_mean, tr_X_std= norm_params_4_val
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - tr_X_mean
                self.X[i] = np.divide(self.X[i], tr_X_std)            
                # self.Y[i] = self.Y[i] - tr_Y_mean
                # self.Y[i] = np.divide(self.Y[i], tr_Y_std)
        # store actions across realizations
        # TODO: try action normalization




    def get_src_stats(self):
        return (self.X_mean, self.X_std)


    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        _, _, _, _, _, _, success, target_pos, _, flow_dir, rzn = self.dataset[idx]
        actions = self.Y[idx]
        traj_len = len(actions)
        env_coef_seq = self.X[idx, :self.context_len, :] # X.shape = (B(r), ETA, coefs+obs_tok)
        padding_len = None
        if traj_len > self.context_len:
            # TODO: correcly write if condition
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            # states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # # NOTE: add extra padde
            # states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            try:
                actions = torch.from_numpy(actions[si : si + self.context_len + 1])
            except:
                actions = torch.cat([actions,
                                    torch.zeros(([1] + list(actions.shape[1:])),
                                    dtype=actions.dtype)],
                                    dim=0)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            print(f" entering if condition - should not happen for basic cases")
            print(f"actions.shape = {actions.shape}")
            print(f"traj_len = {traj_len}")
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(actions)

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [padding_len+1] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        
        target_state = torch.tensor(target_pos)


        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx, flow_dir, rzn

class create_action_dataset_v2_aug(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        norm_params_4_val=None):
        """
        Different from v1: 
            - deals with a combined dataset made from multiple
              datasets with varying obstacle configs
            - dataset is cleaned.

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """

        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = dataset
        # self.X = []
        # for item in dataset:
        #     if len(item[3]) <= 120:
        #         a = np.concatenate([item[3],np.zeros((120-len(item[3]),3))],axis=0)
        #         self.X.append(np.concatenate([item[0],item[1],a],axis=-1).astype(np.float32))
        #     else:
        #         a = item[3][0:120]
        #         self.X.append(np.concatenate([item[0],item[1],a],axis=-1).astype(np.float32))
                
        
        self.X = np.array([np.concatenate((item[0], item[1]), axis=-1) for item in self.dataset])
        # self.X = np.array([item[0] for item in self.dataset]) # Uncomment for DOLS
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)
        eps = 2e-15

        # extract actions (tgt) and scale them to range [0,1)
        # TXY_MEAN = np.array([35.34795 , 41.64394 , 44.63251 ])
        # TXY_STD = np.array([20.757399, 14.942655, 18.891838])
        self.txy = [(item[3][0:-1] - TXY_MEAN) / TXY_STD for item in self.dataset]
        self.a = [item[2]/(2*np.pi) for item in self.dataset]
        self.Y = [np.concatenate([self.a[i],self.txy[i]],axis=-1) for i in range(len(self.a))]

        # normalzise
        if norm_params_4_val == None:
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - self.X_mean
                self.X[i] = np.divide(self.X[i], self.X_std + eps)
                # self.Y[i] = self.Y[i] - self.Y_mean
                # self.Y[i] = np.divide(self.Y[i], self.Y_std)
        else:
            tr_X_mean, tr_X_std= norm_params_4_val
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - tr_X_mean
                self.X[i] = np.divide(self.X[i], tr_X_std + eps)            
                # self.Y[i] = self.Y[i] - tr_Y_mean
                # self.Y[i] = np.divide(self.Y[i], tr_Y_std)
        # store actions across realizations
        # TODO: try action normalization




    def get_src_stats(self):
        return (self.X_mean, self.X_std)


    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        _, _, _, _, _, _, success, target_pos, _, flow_dir, rzn = self.dataset[idx]
        actions = self.Y[idx]
        traj_len = len(actions)
        # env_coef_seq = self.X[idx, :self.context_len, :] # X.shape = (B(r), ETA, coefs+obs_tok+state)
        env_coef_seq = self.X[idx] # env_ceof_seq.shape =  120, coefs+obs_tok+state
       
        padding_len = None
        
        if traj_len > self.context_len:
            # TODO: correcly write if condition
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            # states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # # NOTE: add extra padde
            # states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            try:
                actions = torch.from_numpy(actions[si : si + self.context_len + 1])
            except:
                actions = torch.cat([actions,
                                    torch.zeros(([1] + list(actions.shape[1:])),
                                    dtype=actions.dtype)],
                                    dim=0)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            print(f" entering if condition - should not happen for basic cases")
            print(f"actions.shape = {actions.shape}")
            print(f"traj_len = {traj_len}")
            raise ValueError("case not coded for")
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(actions.astype(np.float32))

            env_coef_seq = torch.from_numpy(env_coef_seq)
            env_coef_seq = torch.cat([env_coef_seq,
                                      torch.zeros(([self.context_len-120]+list(env_coef_seq.shape[1:])))
                                      ], dim=0)

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [padding_len+1] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        
        target_state = torch.tensor(target_pos)


        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx, flow_dir, rzn


def load_velocity(flow_dir):
    flow_dir = Path(flow_dir)
    flow_dir = flow_dir.parent
    flow_dir = flow_dir.parent
    flow_dir = str(flow_dir)
    all_u_mat = np.load(flow_dir +'/all_u_mat.npy')
    all_ui_mat = np.load(flow_dir +'/all_ui_mat.npy')
    all_v_mat = np.load(flow_dir +'/all_v_mat.npy' )
    all_vi_mat = np.load(flow_dir +'/all_vi_mat.npy')
    all_Yi = np.load(flow_dir +'/all_Yi.npy' )
    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
    return vel_field_data

def extract_velocity(vel_field_data, t, rzn):
    nmodes =  vel_field_data[2].shape[1]
    vx = vel_field_data[0][t,:,:].copy()
    vy = vel_field_data[1][t,:,:].copy()

    for m in range(nmodes):
        vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

    return np.stack([vx,vy], axis=0)

def preprocessing_for_mae(vx_vy_list):
    vx_vy_tensor = torch.tensor(vx_vy_list)
    vx_vy_tensor = F.interpolate(vx_vy_tensor, size=(256, 256), mode='bilinear', align_corners=False)     #120,2,256,256
    return vx_vy_tensor    
    
def convert_vel_to_image(im, image_size):
    # TODO: do it for all images in one go
    sz = image_size
    im_tensor = torch.tensor(im)
    # im_tensor = im_tensor.permute(2,0,1)    # 2,100,100
    im_tensor = im_tensor.unsqueeze(0)  # Add batch dimension
    im_tensor = F.interpolate(im_tensor, size=(sz, sz), mode='bilinear', align_corners=False)     # 1,2,256,256
    im_tensor = im_tensor.squeeze(0)  # Remove batch dimension              
    
    return im_tensor   

class create_action_dataset_v3(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        mae,
                        norm_params_4_val=None):
        """
        Different from v2:  (see v2 docstring for difference wrt v1)
            - env_coef_seq return variable is now representation vectors 
             obtained from an MAE encoder
            - Yi_r and obs)r are dummy objects in the dataset since 
                we have combined representations from the MAE encoder
        

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """
        self.std_eps = 1e-6

        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.mae = mae
        self.dataset = dataset
        # extract actions (tgt) and scale them to range [0,1)
        self.Y = [item[2]/(2*np.pi) for item in self.dataset]

      
    # def get_src_stats(self):
    #     return (self.X_mean, self.X_std)

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        env_coef_seq, _, _, _, _, _, success, target_pos, _, flow_dir, rzn = self.dataset[idx]
        actions = self.Y[idx]
        traj_len = len(actions)
        # self.X = np.array([self.extract_latent_rep(item[-2],item[-1]).cpu().numpy() for item in self.dataset]) # Old version
        # env_coef_seq = self.X[idx, :self.context_len, :] # X.shape = (B(r), ETA, coefs+obs_tok) # Old version
        # env_coef_seq = np.array([self.extract_latent_rep(flow_dir,rzn).cpu().numpy()])
        # env_coef_seq = self.extract_latent_rep(flow_dir,rzn)
        
        # Layer normalization
        env_coef_seq_mean = torch.mean(env_coef_seq, axis=-1).reshape(-1,1)
        env_coef_seq_std = torch.std(env_coef_seq, axis=-1).reshape(-1,1)
        env_coef_seq_std[torch.where(env_coef_seq_std==0)] = self.std_eps

        env_coef_seq = env_coef_seq - env_coef_seq_mean
        env_coef_seq = torch.divide(env_coef_seq, env_coef_seq_std)
        
        
        padding_len = None
        if traj_len > self.context_len:
            # TODO: correcly write if condition
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            # states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # # NOTE: add extra padde
            # states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            try:
                actions = torch.from_numpy(actions[si : si + self.context_len + 1])
            except:
                actions = torch.cat([actions,
                                    torch.zeros(([1] + list(actions.shape[1:])),
                                    dtype=actions.dtype)],
                                    dim=0)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            print(f" entering if condition - should not happen for basic cases")
            print(f"actions.shape = {actions.shape}")
            print(f"traj_len = {traj_len}")
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(actions)

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [padding_len+1] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        
        target_state = torch.tensor(target_pos)


        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx, flow_dir, rzn


class create_action_dataset_v4(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        mae,
                        norm_params_4_val=None):
        """
        Different from v3:  (see v3 docstring for difference wrt v2)
            - returns start and end locations as part of input seqence        

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """
        self.std_eps = 1e-6

        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.mae = mae
        self.dataset = dataset
        # extract actions (tgt) and scale them to range [0,1)
        self.Y = [item[2]/(2*np.pi) for item in self.dataset]


    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        env_coef_seq, _, _, _, _, _, success, target_pos, start_pos, flow_dir, rzn = self.dataset[idx]
        actions = self.Y[idx]
        traj_len = len(actions)
        # self.X = np.array([self.extract_latent_rep(item[-2],item[-1]).cpu().numpy() for item in self.dataset]) # Old version
        # env_coef_seq = self.X[idx, :self.context_len, :] # X.shape = (B(r), ETA, coefs+obs_tok) # Old version
        # env_coef_seq = np.array([self.extract_latent_rep(flow_dir,rzn).cpu().numpy()])
        # env_coef_seq = self.extract_latent_rep(flow_dir,rzn)
        # nT, dim = 
        encoder_input = torch.zeros((1,env_coef_seq.shape[-1]))
        # TODO: normalized pos (hardcoded) has different scale than representatons
        encoder_input[0,0:2] = torch.tensor(start_pos)/100.
        encoder_input[0,2:4] = torch.tensor(target_pos)/100.
        encoder_input = torch.cat([encoder_input, env_coef_seq], axis=-2)
        # Layer normalization
        encoder_input_mean = torch.mean(encoder_input, axis=-1).reshape(-1,1)
        encoder_input_std = torch.std(encoder_input, axis=-1).reshape(-1,1)
        encoder_input_std[torch.where(encoder_input_std==0)] = self.std_eps

        encoder_input = encoder_input - encoder_input_mean
        encoder_input = torch.divide(encoder_input, encoder_input_std)
        
        
        padding_len = None
        if traj_len > self.context_len:
            # TODO: correcly write if condition
          
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(actions)

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [padding_len+1] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        
        target_state = torch.tensor(target_pos)


        return  timesteps, actions, traj_mask, target_state, encoder_input, traj_len, idx, flow_dir, rzn
    
    
class create_action_dataset_v5(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        pad_chan3 = False,
                        norm_params_4_val=None):
        """
        Different from v4:  (see v4 docstring for difference wrt v3)
            -       

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """
        self.std_eps = 1e-6

        self.context_len = context_len
        self.n_trajs = len(dataset)
        # self.mae = mae
        self.dataset = dataset
        # extract actions (tgt) and scale them to range [0,1)
        self.Y = [item[2]/(2*np.pi) for item in self.dataset]


    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        env_coef_seq, _, _, _, _, _, success, target_pos, start_pos, flow_dir, rzn = self.dataset[idx]
        actions = self.Y[idx]
        traj_len = len(actions)

        vel_field_data = load_velocity(flow_dir)
        #TODO: remove hardcoded 256 image size and take input from config
        velocities_r = [convert_vel_to_image(extract_velocity(vel_field_data,t,rzn),224) for t in range(env_coef_seq.shape[0]) ]
        velocities_r = torch.stack(velocities_r, axis=0)  # shape (post): [120, 2, 256, 256]

        """
        TODO: need to add location and normalization in custom model after representations.
        # encoder_input = torch.zeros((1,env_coef_seq.shape[-1])) #
        # # TODO: normalized pos (hardcoded) has different scale than representatons
        # encoder_input[0,0:2] = torch.tensor(start_pos)/100.
        # encoder_input[0,2:4] = torch.tensor(target_pos)/100.
        # encoder_input = torch.cat([encoder_input, env_coef_seq], axis=-2)
        # # Layer normalization
        # encoder_input_mean = torch.mean(encoder_input, axis=-1).reshape(-1,1)
        # encoder_input_std = torch.std(encoder_input, axis=-1).reshape(-1,1)
        # encoder_input_std[torch.where(encoder_input_std==0)] = self.std_eps

        # encoder_input = encoder_input - encoder_input_mean
        # encoder_input = torch.divide(encoder_input, encoder_input_std)
        """
        
        padding_len = None
        if traj_len > self.context_len:
            # TODO: correcly write if condition
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(actions)

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [padding_len+1] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        
        target_state = torch.tensor(target_pos)
        start_state = torch.tensor(start_pos)
        loc_tensor = torch.cat([start_state, target_state], dim=0)

        return  timesteps, actions, traj_mask, loc_tensor, velocities_r, traj_len, idx, flow_dir, rzn


class create_env_emb_dataset(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        pad_chan3 = False,
                        norm_mode="BN",
                        norm_params_4_val=None,
                        ):
        """
        dataset of env embeddings
            -       
        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 
        """
        
        self.std_eps = 1e-6
        self.context_len = context_len
        self.n_samples = len(dataset)
        self.dataset = dataset

        ### Commented by Shubham - Testing Perlin fields (13/12/2023)
        # emb_list_tensors = [item[0] for item in dataset]
        # self.X = torch.stack(emb_list_tensors)        
        
        # if norm_mode == "BN":
        #     if norm_params_4_val == None:
        #         self.X_mean = self.X.mean(axis=0)
        #         self.X_std = self.X.std(axis=0)
        #         for i in range(len(self.X)):
        #             self.X[i] = self.X[i] - self.X_mean
        #             self.X[i] = np.divide(self.X[i], self.X_std + self.std_eps)

        #     else:
        #         tr_X_mean, tr_X_std= norm_params_4_val
        #         for i in range(len(self.X)):
        #             self.X[i] = self.X[i] - tr_X_mean
        #             self.X[i] = np.divide(self.X[i], tr_X_std + self.std_eps)    

        # if norm_mode == "LN":
        #     for i in range(len(self.X)):
        #         self.X[i] = self.X[i] - self.X[i].mean(axis=-1)
        #         self.X[i] = np.divide(self.X[i], self.X[i].std(axis=-1) + self.std_eps)
                
                
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        env_reps, _, _, _,_, _, _, _,_,_, flow_dir, rzn = self.dataset[idx]
        # for cases where env_reps comes from mode coeffecients. 
        if not torch.is_tensor(env_reps):
            env_reps = torch.tensor(env_reps)
        
        padding_len = None
        traj_len = len(env_reps)
        if traj_len > self.context_len:
            # TODO: correcly write if condition
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
        
            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            env_reps = torch.cat([env_reps, # shape (nenv_reps, 1)
                                # [padding_len+1] is seq_len axis, list(env_reps.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(env_reps.shape[1:])), #[4]+[1]=[4,1]
                                dtype=env_reps.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)

        return  timesteps, env_reps, traj_mask, idx, flow_dir, rzn



class create_envEnc_dtDec_dataset(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        nT, # contextl len
                        rtg_scale,
                        autoenc = None ,
                        encoding_type = 'tae', #
                        norm_params_4_val=None,
                        LN=False,
                        device = 'cpu'):
        """
        derived from create_action_dataset_v4
        for creating dataset in train_pptt.py
          env_emb encoder
          decTransformer decoder

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, rewards_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """
        if encoding_type == 'tae':
            print(" ----- \n Using *TAE* to create embeddings \n ---")
            self.extractor = ExtractRep_tae(None, autoenc) 
            self.autoenc = autoenc.to(device)
        elif encoding_type == 'mae':
            print(" ----- \n Using *MAE* to create embeddings \n ---")
            self.extractor = ExtractRep_mae(None, autoenc)
            self.autoenc = autoenc.to(device)
        elif encoding_type == 'Yis':    
            print(" ----- \n Using *Yis* as embeddings \n ---")
            print(f"Yis or DO coeffs are already in gathered dataset")
            self.extractor = None
        else:
            raise ValueError("invalid extractor class")
        
        self.nT = nT
        self.enc_max_len = nT
        self.dec_max_len = 3*(nT) + 1
        self.LN = LN
        self.std_eps = 1e-6
        self.rtg_scale = rtg_scale
        self.n_trajs = len(dataset)
        self.dataset = dataset
        self.device = device

        # TODO: take a random subset of dataset to calculate statistics
        #        It will help in case dataset is too large
        if norm_params_4_val == None:
            states = [item[3] for item in self.dataset]
            states = np.concatenate(states, axis=0)
            self.states_mean = np.mean(states, axis=0)
            self.states_std = np.std(states, axis=0)
        else:
            self.states_mean, self.states_std = norm_params_4_val


    def __len__(self):
        return len(self.dataset)
        
    def get_states_stats(self):
        return self.states_mean, self.states_std
    
    # @execution_time #0.085  secs
    def __getitem__(self, idx):
        # Yi_r, obs_r, actions_r, states_r, timesteps_r, rewards_r, dones_r, success_r, 
        # target_pos_r, start_pos_r, flow_dir, rzn 
                     
        Yi_r, _, actions, states,  timesteps, rews, dones, success, trg_pos, st_pos, flow_dir, rzn = self.dataset[idx]
        
        trg_pos = np.array(trg_pos, dtype=np.float32)
        # scale actions to [0,1]    
        actions = actions / (2*np.pi)
        states = (states - self.states_mean) / (self.states_std + self.std_eps)
        # normalize trg position (xy). states_mean is normalized (txy)
        trg_pos = (trg_pos - self.states_mean[1:]) / (self.states_std[1:] + self.std_eps)
        # TODO: verify corrrectness, 0 at end
        rtg = np.flip(np.cumsum(np.flip(rews))) / self.rtg_scale
        n = len(actions)
        # dec_io_seq_len = 3*n + 1 #  x0 - {RSA}*n - 0
        
        # encoder input
        # start_time = time.time()
        if self.extractor == None:
            encoder_input = Yi_r
            encoder_input = torch.tensor(encoder_input, dtype=torch.float32).to(self.device)
        else:
            encoder_input = self.extractor.extract_latent_rep(self.autoenc, flow_dir,rzn,return_type='gpu')  #(120,d=768)
        # end_time = time.time()
        # print(f"in datasetcreator, mae extraction time = {(end_time - start_time)/60} mins")
        # Layer normalization
        # TODO: Try with and without
        if self.LN:
            encoder_input_mean = torch.mean(encoder_input, axis=-1).reshape(-1,1)
            encoder_input_std = torch.std(encoder_input, axis=-1).reshape(-1,1)
            encoder_input_std[torch.where(encoder_input_std==0)] = self.std_eps

            encoder_input = encoder_input - encoder_input_mean
            encoder_input = torch.divide(encoder_input, encoder_input_std)  
        
        if n > self.nT:
            # TODO: correcly write if condition
            raise ValueError(f"Case not accounted for: dec_input_seq () > dec_len () ")
        else:
            e_pad_len = self.nT - encoder_input.shape[0] 
            encoder_input = torch.cat([encoder_input, # shape (120, d)
                                # [dec_pad_len] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([e_pad_len] + list(encoder_input.shape[1:])), #[4]+[1]=[4,1]
                                dtype=encoder_input.dtype, device=encoder_input.device)],
                                dim=0)
            encoder_input_mask = create_padding_mask(self.nT,e_pad_len)
            
            a_pad_len = self.nT - n
            # padding with zeros
            actions = torch.from_numpy(actions)
            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [dec_pad_len] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([a_pad_len] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                                dim=0)
            # actons.shape = (121,1)
            a_input_mask = create_padding_mask(self.nT,e_pad_len)

            # s_pad_len to make post padded s,r,a sequences of the same length
            # since states list has 1 extra element compared to r and a
            s_pad_len = a_pad_len - 1 if a_pad_len > 0 else 0
            states = torch.from_numpy(states)
            states = torch.cat([states, # shape (nactions, 1)
                                # [dec_pad_len] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([s_pad_len] + list(states.shape[1:])), #[4]+[1]=[4,1]
                                dtype=states.dtype)],
                               dim=0)    

            rtg = torch.from_numpy(rtg).unsqueeze(-1)
            rtg = torch.cat([rtg, # shape (nactions, 1)
                                # [dec_pad_len] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([a_pad_len] + list(rtg.shape[1:])), #[4]+[1]=[4,1]
                                dtype=rtg.dtype)],
                               dim=0)                
            timesteps = torch.arange(start=1, end=self.nT+2, step=1)
            
            rtg_mask = create_padding_mask(self.nT, a_pad_len)
            # +1 to pad the terminal state as it will not predict any action
            states_mask = create_padding_mask(self.nT, s_pad_len+1) 
            action_mask = create_padding_mask(self.nT, a_pad_len)
 
        final_pos = torch.tensor(trg_pos)
        tgt_padding_mask = assemble_trsa_masks(self.dec_max_len, rtg_mask, states_mask, action_mask)
        
        return  timesteps, actions, states, rtg, action_mask, final_pos, encoder_input, tgt_padding_mask, n, idx, flow_dir, rzn


class cgw_trajec_dataset(Dataset):
    def __init__(self, trajectories, context_len, rtg_scale, state_dim=None):
        """
        trajectories; 
        context_len:
        rtg_scale:
        state_dim: if = 5, then state is 'txyuv'
        """
  
        self.context_len = context_len
        # TODO: Change if not pkl file

        self.trajectories = trajectories
        total_n_trajs = len(trajectories)
        self.trajectories = [traj for traj in self.trajectories if traj['done']]
        print(f"\n Making dataset out of successful trajectories. \n \
                No. of successful trajs / total trajs = {len(self.trajectories)} / {total_n_trajs} \n")

        min_len = 10**6     #high intiialization to update later
        states = []
        # print(f" ****** Verify: len of first traj (in cgw)= {len(self.trajectories[0]['states'])} ")
        # print(f" ****** Verify: first traj (in cgw)= {self.trajectories[0]['states']} ")

        for traj in self.trajectories:
            traj_len = traj['states'].shape[0]
            min_len =  min(min_len, traj_len)
            states.append(traj['states'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        # normalize states
        for traj in self.trajectories:
            traj['states'] = (traj['states'] - self.state_mean) / self.state_std
        # print(f" ****** Verify: first normalized traj (in cgw)= {self.trajectories[0]['states']} ")



    def get_state_stats(self):
        return self.state_mean, self.state_std

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['states'].shape[0]
        # print(f"****** VERIFY: traj_len = {traj_len}")
        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['states'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)
            target_state = torch.from_numpy(traj['target_pos'])

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['states'])
            # print(f" in cwg: states.shape = {states.shape}")

            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)
            # print(f" in cwg: states.shape = {states.shape}")

            # sys.exit()
            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)
            target_state = torch.from_numpy(traj['target_pos'])

        # print(f"### verify: target_state = {target_state}")
        dummy_t = float(int(0.8*self.context_len)) # time stamp for target state.
        target_state = np.insert(target_state,0,dummy_t,axis=0)
        # print(f"### verify: target_state = {target_state}")
        return  timesteps, states, actions, returns_to_go, traj_mask, target_state

    
def assemble_trsa_masks(tgt_seq_len: int, rmask, smask, amask):
    # shape of rs, ss, as shoudl be sam
    T = tgt_seq_len
    tgt_padding_mask = torch.zeros((T,)).type(torch.bool)
    tgt_padding_mask[0] = False
    tgt_padding_mask[1:1+3*T:3] = rmask
    tgt_padding_mask[2:2+3*T:3] = smask
    tgt_padding_mask[3:3+3*T:3] = amask
        
    return tgt_padding_mask

def convert_angle_to_vectors(angle_seq, device='cpu'):
    """
    angle_seq: shape = (n,1)
    """
    vec_seq = torch.zeros(angle_seq.shape[0],2).to(device)
    vec_seq[:,0] = torch.cos(angle_seq[:,0]*2*torch.pi)
    vec_seq[:,1] = torch.sin(angle_seq[:,0]*2*torch.pi)

    return vec_seq
"""
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
# mask for sz = 3. It is an ADDITIVE mask
    # [[0., -inf, -inf],
    #         [0., 0., -inf],
    #         [0., 0., 0.]]

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, padding_len, device, extra_tokens=0):
    src_seq_len = src.shape[1] + extra_tokens #(BS, context_len, hdim)
    tgt_seq_len = tgt.shape[1]
    # assert(src_seq_len==tgt_seq_len)
    batch_size = src.shape[0]
    # print(f" in create mask: src_seq_len.shape = {src_seq_len}")
    # print(f" in create mask: tgt_seq_len.shape = {tgt_seq_len}")

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # src_padding_mask = (src == src).transpose(0, 1) # (context_len, BS, hdim)
    
    src_padding_mask = torch.zeros((batch_size,src_seq_len),device=device).type(torch.bool)

    tgt_padding_mask = torch.zeros((batch_size,src_seq_len),device=device).type(torch.bool)
    for i in range(batch_size):
        tgt_padding_mask[i,:padding_len[i]] =  True
    # print(f" in create mask: tgt_padding_mask.shape = {tgt_padding_mask.shape}")
    # print(f" in create mask: tgt_padding_mask = {(tgt_padding_mask.type(torch.int))}")
    # print(f" verify sum {torch.sum(tgt_padding_mask, axis=1)}")
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def create_padding_mask(seq_len: int, pad_len: int):
    """
    creates a mask that 
    pads last pad_len items of seq_len sized vector
    Note: no batch dim. to be used in create_x_dataset_funcition._get_item()
    """
    padding_mask = torch.zeros((seq_len,)).type(torch.bool)
    padding_mask[-pad_len:] = True

    return padding_mask


def plot_vel_field(env,t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
    # Make modes the last axis T m H W
    Ui = np.transpose(env.Ui,(0,2,3,1))
    Vi = np.transpose(env.Vi,(0,2,3,1))
    # print(f"**** verify: t = {t}")
    vx_grid = env.U[t,:,:] + np.dot(Ui[t,:,:,:],env.Yi[t,r,:])
    vy_grid = env.V[t,:,:] + np.dot(Vi[t,:,:,:],env.Yi[t,r,:])
    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    Xs = np.arange(0,env.xlim) + (env.dxy/2)
    Ys = np.arange(0,env.ylim) + (env.dxy/2)
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)

def plot_vel_field_perlin(env,t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
    # Make modes the last axis
    # Ui = np.transpose(self.env.Ui,(0,2,3,1))
    # Vi = np.transpose(self.env.Vi,(0,2,3,1))
    vx_grid = env.U[t,:,:] #+ np.dot(Ui[t,:,:,:],self.env.Yi[t,r,:])
    vy_grid = env.V[t,:,:] #+ np.dot(Vi[t,:,:,:],self.env.Yi[t,r,:])
    # vx_grid = np.flipud(vx_grid)
    # vy_grid = np.flipud(vy_grid)
    Xs = np.arange(0,env.xlim) + (env.dxy/2)
    Ys = np.arange(0,env.ylim) + (env.dxy/2)
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    # v_mag_grid[np.where(obs_mask==1)]=np.nan
    # plt.contourf(X, Y, v_mag_grid, alpha = 0.9, zorder = -1e5, colors='white')
    # ax.set_facecolor("grey") 
    plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)
    # ax = plt.gca()
    # ax.set_facecolor('grey')
    # return im


def denormalize(txy_norm,tr_stats):
    mean, std = tr_stats
    return (txy_norm*std) + mean

def discrete_frechet_distance(P, Q):
    N = P.shape[0]
    M = Q.shape[0]
    inf = float('inf')
    d = torch.zeros((N, M), dtype=torch.float32)
    d[0, 0] = torch.norm(P[0] - Q[0], 2)
    d[1:, 0] = inf
    d[0, 1:] = inf
        
    for i in range(1, N):
        for j in range(1, M):
            d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + torch.norm(P[i] - Q[j], 2)
    return d[N - 1, M - 1]

def DOLS_obstacle():
    s=15
    return plt.Circle([4.5*s, 3*s], 0.5*s, color='k', alpha=0.3)


    
def visualize_output(preds_list, 
                        path_lens,
                        iter_i = 0, 
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        at_time=None,
                        color_by_time=True,
                        plot_flow=True,
                        wandb_suffix="",
                        model_name = "",
                        ground_truth_states = None, #for debugging or comparison
                        preds_list_from_target_actions = None, # for denugging or comparison
                        ):

    # colours =['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'deeppink', 'cyan']
    colours = ['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'deeppink', 'cyan', 
           'brown', 'turquoise', 'pink', 'gray', 'navy', 'plum', 'lime', 'salmon', 
           'tan', 'skyblue', 'orchid', 'slateblue']

    colour_list=[]
      
    print(f"path_lens = {path_lens}")
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + model_name + "_pred_on_unseen" + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.title(f"Policy execution after {iter_i} epochs")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(preds_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    if color_by_time:
        # t_dones = []
        # for preds in preds_list:
        #     t_dones.append(op_traj_dict['t_done'])
        vmin = min(path_lens) if len(path_lens)>0 else 0
        vmax = max(path_lens) if len(path_lens)>0 else 70
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)


    for idx,traj in enumerate(preds_list):
        if idx in traj_idx:
            random_color = colours.pop()
            colour_list.append(random_color)
            if idx%100==0:
                print(idx)
            states = preds_list[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx] #TODO: change 
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                print(f"{states.shape}, {mean.shape}")
                states = (states*std) + mean
        
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=random_color, alpha = 0.5) #, color=scalarMap.to_rgba(t_done), alpha=0.5)
            else:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=2, marker='^', color='red', label='Predicted_states')

            # Plot policy at visites states
            _, nstates,_ = states.shape
            # if plot_policy:
            #     for i in range(nstates):
            #         plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))
            
    for idx,traj in enumerate(ground_truth_states):
        if idx in traj_idx:
            if idx%100==0:
                print(idx)
            states = ground_truth_states[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx] #TODO: change 
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                print(f"{states.shape}, {mean.shape}")
                states = (states*std) + mean
        
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=colour_list[idx], alpha = 0.5) #, color=scalarMap.to_rgba(t_done), alpha=0.5)
            else:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=2, marker='o', color='green', label='Ground_truth_states')

            # Plot policy at visites states
            _, nstates,_ = states.shape
            # if plot_policy:
            #     for i in range(nstates):
            #         plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))          
            
    # for idx,traj in enumerate(preds_list_from_target_actions):
    #     if idx in traj_idx:
    #         if idx%100==0:
    #             print(idx)
    #         states = preds_list_from_target_actions[idx]
    #         states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
    #         states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
    #         t_done = path_lens[idx] #TODO: change 
    #         # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
    #         if at_time != None:
    #             assert(at_time >= 1), f"Can only plot at_time >= 1 only"
    #             # if at_time > t_done, just plot for t_done
    #             at_time = min(at_time, t_done)
    #         else:
    #             at_time = t_done

    #         if stats!=None:
    #             mean, std = stats
    #             print(f"{states.shape}, {mean.shape}")
    #             states = (states*std) + mean
        
    #         # shape: (eval_batch_size, max_test_ep_len, state_dim)
    #         if color_by_time:
    #             plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=scalarMap.to_rgba(t_done), alpha=0.5)
    #         else:
    #             plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

    #         if show_scatter:
    #             plt.scatter(states[0,:,1], states[0,:,2],s=1, c='green',label='Preds_list_from_target_actions')

    #         # Plot policy at visites states
    #         _, nstates,_ = states.shape
    #         # if plot_policy:
    #         #     for i in range(nstates):
    #         #         plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))
    # # plt.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize='small')
    
    if color_by_time:
        cbar = plt.colorbar(scalarMap, label="Arrival Time")
    # TODO: remove hardcode
    plt.xlim([0., 100.])
    plt.ylim([0., 100.])
    # plot target area and set limits
    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        # obstacle = DOLS_obstacle()
        # ax.add_patch(obstacle)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
        if plot_flow and at_time!=None:
            t = at_time
            plot_vel_field_perlin(env,t)
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"pred_traj_fig_"+wandb_suffix: wandb.Image(fname)})


    return colour_list

def visualize_actions(model_save_dir, flow_dir, colours, train_set_actions_preds, 
                        train_set_actions_ground,
                        log_wandb=True, 
                        ):
    # try:
    #     os.mkdir(join(model_save_dir, 'figs'))
    # except:
    #     print('Figs dir exists')

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()

    for idx,traj in enumerate(train_set_actions_preds):
        plt.cla()
        plt.plot(train_set_actions_ground[idx][0], color=colours[idx], label='ground_truth_actions')
        plt.plot(train_set_actions_preds[idx][0], linestyle='dotted', color='red', label='preds_actions')
        plt.legend(loc='upper right', fontsize='small')
        flow_dir_ = np.int32(flow_dir[idx].split('/')[-1].split('_')[-1])
        plt.title(f'Flow_dir: {flow_dir_}')
        plt.savefig(join(model_save_dir, f'actions_plot_{idx}.png'), dpi=300)
        
    fig, axs = plt.subplots(5, 4, figsize=(12, 10))
    for i in range(len(train_set_actions_preds)):

        image_path = join(model_save_dir, f"actions_plot_{i}.png")
        image = Image.open(image_path)
        row = i // 4  
        col = i % 4   
        axs[row, col].imshow(image)
        image.close()

    fname=join(model_save_dir, "collated.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"preds_vs_ground_actions": wandb.Image(fname)})

    return fig

def visualize_actions_flow(model_save_dir,
                        train_op_traj_dict_list, 
                        sim_train_op_traj_dict_list,
                        src_stats=None,
                        log_wandb=False, 
                        ):


    colours = ['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'deeppink', 'cyan', 
           'brown', 'turquoise', 'pink', 'gray', 'navy', 'plum', 'lime', 'salmon', 
           'tan', 'skyblue', 'orchid', 'slateblue']
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()

    train_set_txy_preds = [d['states']*src_stats[1] + src_stats[0] for d in train_op_traj_dict_list]
    path_lens = [d['n_tsteps'] for d in train_op_traj_dict_list]
    ground_truth_states_train = [d['ground_truth_states']*src_stats[1] + src_stats[0] for d in sim_train_op_traj_dict_list] 
    flow_dir = [d['flow_dir'] for d in train_op_traj_dict_list]
    train_set_actions_preds = [d['actions'] for d in train_op_traj_dict_list]
    train_set_actions_ground = [d['actions'] for d in sim_train_op_traj_dict_list]  

    for idx, traj in enumerate(train_set_txy_preds):
        plt.cla()
        env = setup_env(flow_dir[idx])
        visualize_output_old(flow_dir[idx],
                            idx,
                            model_save_dir,
                            [train_set_txy_preds[idx]], 
                            [path_lens[idx]],
                            iter_i = 0, 
                            stats=None, 
                            env=env, 
                            log_wandb=False, 
                            plot_policy=False,
                            traj_idx=None,      #None=all, list of rzn_ids []
                            show_scatter=True,
                            at_time=59,
                            color_by_time=False, #TODO: fix tdone issue in src_utils
                            plot_flow=True,
                            wandb_suffix="val",
                            ground_truth_states=[ground_truth_states_train[idx]]
                            )
        
        fig = plt.figure()
        plt.cla()
        ax = plt.gca()
        plt.plot(train_set_actions_ground[idx][0], color=colours[idx], label='ground_truth_actions')
        plt.plot(train_set_actions_preds[idx][0], linestyle='dotted', color='red', label='preds_actions')
        plt.legend(loc='upper right', fontsize='small')
        flow_dir_ = np.int32(flow_dir[idx].split('/')[-1].split('_')[-1])
        plt.title(f'Flow_dir: {flow_dir_}')
        plt.savefig(join(model_save_dir, f'actions_plot_{idx}.png'), dpi=300)
        

    fig, axs = plt.subplots(len(train_set_txy_preds), 3, figsize=(4, 25))
    for i in range(len(train_set_txy_preds)):
        actions = Image.open(join(model_save_dir, f"actions_plot_{i}.png"))
        flow_with_traj = Image.open(join(model_save_dir, f"flow_with_traj_{i}.png"))
        try:
            flow = Image.open(join(flow_dir[i], 'perlin_figs/perlin_field_0.png'))
        except:
            flow = Image.open(join(flow_dir[i], 'perlin_figs_60/perlin_field_0.png'))
        axs[i, 0].imshow(actions)
        axs[i, 0].axis('off')
        axs[i, 1].imshow(flow_with_traj)
        axs[i, 1].axis('off')
        axs[i, 2].imshow(flow)
        axs[i, 2].axis('off')

    fname=join(model_save_dir, "collated_actions_trajs_flow.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"preds_vs_ground_actions": wandb.Image(fname)})

    return fig


def visualize_output_old(flow_dir,
                        idx_,
                        model_save_dir,
                        preds_list, 
                        path_lens,
                        iter_i = 0, 
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        at_time=None,
                        color_by_time=True,
                        plot_flow=True,
                        wandb_suffix="",
                        model_name = "",
                        ground_truth_states=None,
                        ):

    print(f"path_lens = {path_lens}")
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + model_name + "_pred_on_unseen" + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.title(f"Policy execution after {iter_i} epochs")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(preds_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    if color_by_time:
        # t_dones = []
        # for preds in preds_list:
        #     t_dones.append(op_traj_dict['t_done'])
        vmin = min(path_lens) if len(path_lens)>0 else 0
        vmax = max(path_lens) if len(path_lens)>0 else 70
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)


    for idx,traj in enumerate(preds_list):
        if idx in traj_idx:
            if idx%100==0:
                print(idx)
            states = preds_list[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx] #TODO: change 
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                print(f"{states.shape}, {mean.shape}")
                states = (states*std) + mean
        
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=scalarMap.to_rgba(t_done), alpha=0.2)
            else:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=1)

            # Plot policy at visites states
            _, nstates,_ = states.shape
            # if plot_policy:
            #     for i in range(nstates):
            #         plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))
    for idx,traj in enumerate(ground_truth_states):
        if idx in traj_idx:
            if idx%100==0:
                print(idx)
            states = ground_truth_states[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx] #TODO: change 
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                print(f"{states.shape}, {mean.shape}")
                states = (states*std) + mean
        
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=scalarMap.to_rgba(t_done), alpha = 0.5) #, color=scalarMap.to_rgba(t_done), alpha=0.5)
            else:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=2, marker='o', color='green', label='Ground_truth_states')

            # Plot policy at visites states
            _, nstates,_ = states.shape
            # if plot_policy:
            #     for i in range(nstates):
            #         plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))          
    
    if color_by_time:
        cbar = plt.colorbar(scalarMap, label="Arrival Time")
    # TODO: remove hardcode
    plt.xlim([0., 100.])
    plt.ylim([0., 100.])
    # plot target area and set limits
    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        # obstacle = DOLS_obstacle()
        # ax.add_patch(obstacle)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
        if plot_flow and at_time!=None:
            t = at_time
            plot_vel_field_perlin(env,t)
    flow_dir=np.int32(flow_dir.split('/')[-1].split('_')[-1])
    plt.title(f'flow_dir: {flow_dir}')    
    plt.savefig(join(model_save_dir, f'flow_with_traj_{idx_}'), dpi=300)

    if log_wandb:
        wandb.log({"pred_traj_fig_"+wandb_suffix: wandb.Image(fname)})


    return fig

def compare_trajectories(tr_op_traj_dict_list,
                            path_lens,
                            iter_i = 0, 
                            stats=None, 
                            env=None, 
                            log_wandb=True, 
                            plot_policy=False,
                            traj_idx=None,      #None=all, list of rzn_ids []
                            show_scatter=False,
                            at_time=None,
                            color_by_time=True,
                            plot_flow=True,
                            wandb_suffix="",
                            ):
    
    tr_set_txy_preds = [d['states'] for d in tr_op_traj_dict_list]
    tr_set_txy_PREDS_ = [d['states_for_action_labels'] for d in tr_op_traj_dict_list]
    actions = [d['actions'] for d in tr_op_traj_dict_list]
    ACTIONS_ = [d["action_labels"] for d in tr_op_traj_dict_list]
    mses = [d['mse'].item() for d in tr_op_traj_dict_list]
    
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "compare_traj" + str(iter_i) + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(tr_set_txy_preds))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    if color_by_time:
        # t_dones = []
        # for preds in preds_list:
        #     t_dones.append(op_traj_dict['t_done'])
        vmin = min(path_lens) if len(path_lens)>0 else 0
        vmax = max(path_lens) if len(path_lens)>0 else 70
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)


    for idx in traj_idx:
        states = tr_set_txy_preds[idx]
        STATES_ = tr_set_txy_PREDS_[idx]
        mse = np.round(mses[idx],6)
        if stats!=None:
            mean, std = stats
            states = (states*std) + mean
            STATES_ = (STATES_*std) + mean
    
        if show_scatter:
            plt.scatter(states[0,:,1], states[0,:,2],s=1, c='r', label=f"id_{idx}: {mse}")
            plt.scatter(STATES_[0,:,1], STATES_[0,:,2],s=1, c='g')

        # Plot policy at visites states
        _, nstates,_ = states.shape
        if plot_policy:
            action = actions[idx]
            ACTION_ =  ACTIONS_[idx]
            for i in range(nstates-1):
                plt.arrow(states[0,i,1], states[0,i,2], np.cos(action[0,i,0]), np.sin(action[0,i,0]))
                plt.arrow(STATES_[0,i,1], STATES_[0,i,2], np.cos(ACTION_[0,i,0]), np.sin(ACTION_[0,i,0]))
    plt.legend()
    if color_by_time:
        cbar = plt.colorbar(scalarMap, label="Arrival Time")
    # TODO: remove hardcode
    plt.xlim([0., 100.])
    plt.ylim([0., 100.])
    # plot target area and set limits
    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"Compare"+wandb_suffix: wandb.Image(fname)})



def simulate_tgt_actions(traj_dataset,
                           env=None,
                           gathered_envs = False,
                           log_wandb=True,
                           wandb_fname='simulate_tgt_actions',
                           plot_flow=True,
                           at_time=119,
                           plot_range=100
                           ):
    
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "simulated_input_traj"  + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    title = "Input_traj_"
    plt.title(title)
    success_count_ = 0
    action_list = [item[2] for item in traj_dataset.dataset]
    rzn_list = [item[-1] for item in traj_dataset.dataset]
    flow_dir_list = [item[-2] for item in traj_dataset.dataset]
    for i in range(plot_range):
        rzn = rzn_list[i]
        flow_dir = flow_dir_list[i]
        if gathered_envs:
            env = setup_env(flow_dir)
        env.set_rzn(rzn)
        env.reset()
        actions =  action_list[i]
        txy_array = np.zeros((121,3))
        txy_array[:,:] = None
        txy_array[0,:] = np.array([0,env.start_pos[0],env.start_pos[1]])
        reached_target_ = False

        for k in range(len(actions)):
            a = actions[k]
            txy, reward ,done, info = env.step(a)
            txy_array[k+1,:] = txy
            if done:
                if reward > 0:
                    reached_target_ = True
                    success_count_ += 1
                break

        plt.plot(txy_array[:,1], txy_array[:,2])
        plt.scatter(txy_array[len(actions),1], txy_array[len(actions),2], s=1.5, zorder=1000)

    plt.xlim([0, 100.])
    plt.ylim([0, 100.])

    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # obstacle = DOLS_obstacle()
        # ax.add_patch(obstacle)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)


        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)    
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({wandb_fname: wandb.Image(fname)})
    plt.cla()
    
def visualize_input(traj_dataset, 
                    stats=None, 
                    env=None, 
                    log_wandb=True,
                    traj_idx=None,      #None=all, list of rzn_ids []
                    wandb_fname='input_traj_fig',
                    info_str='',
                    at_time=None,
                    color_by_time=True,
                    plot_flow=True,
                    data_name=''
                    ):
 
    print(" ---- Visualizing input ---- ")

    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "input_traj" + info_str + data_name + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    title = "Input_traj_" + info_str
    plt.title(title)

    states_list =[item[3] for item in traj_dataset.dataset]


    if stats!=None:
        print("===== Note: rescaling states to original scale for viz (in visualize_input)=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(traj_dataset))]
        # TODO: verify wgy 320 out 400 are here
        # print(f" traj_idx= {traj_idx}")
    if color_by_time:
        t_dones = []
        for idx, traj in enumerate(traj_dataset):
            if idx in traj_idx:
                states = states_list[idx]
                t_done = len(states)  # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for idx, traj in enumerate(traj_dataset):
        if idx in traj_idx:
            states = states_list[idx]
            t_done = t_dones[idx]

            if stats != None:
                mean, std = stats
                states = (states*std) + mean
                # states = states*(traj_mask.reshape(-1,1))

            if color_by_time:
                plt.plot(states[:,1], states[:,2], color=scalarMap.to_rgba(t_done), alpha= 0.2)
            else:
                plt.plot(states[:,1], states[:,2], linewidth=0.1)
            plt.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # print(f"bcords: {states[-1,1], states[-1,2]}")
    plt.xlim([0, 100.])
    plt.ylim([0, 100.])

    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        print("****VERIFY: env.target_pos: ", env.target_pos)
        print(f"**** verify: {len(env.target_pos)}")
        # obstacle = DOLS_obstacle()
        # ax.add_patch(obstacle)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)


        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)    
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({wandb_fname: wandb.Image(fname)})
    #plt.cla()

def see_steplr_trend(step_size: int, num_epochs: int, lr=0.001, final_lr=None, update_inside_epoch = False, len_tr_set=None, gamma=None, show_plot=False):
    if update_inside_epoch:
        nsteps = int(num_epochs*len_tr_set/step_size)
    else:
        nsteps = int(num_epochs/step_size)
        
    lr = lr
    if gamma == None and final_lr != None:
        final_lr = final_lr
        gamma = (final_lr/lr)**(1/nsteps)
        print(f"use gamma = {gamma}")

    if gamma != None and final_lr == None:
        final_lr = lr*gamma**nsteps
        print(f"final_lr= {final_lr}")

    y = np.zeros(num_epochs,)
    for i in range(nsteps):
        y[i*step_size:(i+1)*step_size] = lr*gamma**i
    if show_plot:
        plt.yscale("log")  
        plt.plot(y)
        plt.savefig(f'/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tmp/stepLR.png')
    return gamma, final_lr


def plot_attention_weights(weight_mat, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=False,
                            scale_each_row=False, 
                            causal_mask_used=True,
                            cmap=cm.Reds, 
                            log_wandb = False,
                            fname='attention_heatmap.png',
                            info_string = '',
                            wandb_fname = ''                 
                            ):

    """
    weight_mat: (avgd across heads) weight matrix expected shape = [1-6],32,120,120 or L,B,T,T where T is 3*context_len
    set_idx: sample index of batch
    layer_idx: layer index
    scale_each_row: scales each row INDEPENDENTLY to lie between 0 and 1 for visualization
    """
    plt.cla()
    plt.clf()
    plt.close()
    weights = weight_mat
    shape = weights.shape
    
    # Plot attenetion scores for the ith trajectory/sample among the batch (for training batch)

    
    scale_str = ''
    layer_str = ''
    if average_across_layers:
        layer_str = 'avg'
        weights = np.mean(weights[:,set_idx,:,:],axis=0)
    else:
        layer_str = str(layer_idx)
        weights = weights[layer_idx,set_idx,:,:]

    if scale_each_row:
        scale_str = '_scaled_rows_'
        weights = scale_attention_rows(weights, causal_mask_used=causal_mask_used)


    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_ticks(np.arange(0,120,20))
    ax.yaxis.set_ticks(np.arange(0,120,20))
    shw = ax.imshow(weights, cmap=cmap)
    bar = plt.colorbar(shw)
    bar_fontsize = 12
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    # cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
    bar.set_label("Attention Weights (scaled)", fontsize=bar_fontsize)

    title =  scale_str + info_string  +"_L" + layer_str + "_smp" +  str(set_idx)
    # plt.title(title)
    # plt.figure(figsize=(12,10))
    # ax = sns.heatmap(weights, linewidth=0.05)
    fname=fname + title + ".png"
    plt.savefig(fname, bbox_inches="tight", dpi=600)
    if log_wandb:
        if wandb_fname == None:
            wandb_fname =  "attention map"  
        wandb.log({wandb_fname: wandb.Image(fname)})



def scale_attention_rows(at_weights, causal_mask_used=False):
    """
    at_weights: 2d matrix of attntion weights
    """
    shape = at_weights.shape
    # print(f"**** Verify type= {type(at_weights)}")
    # print(f"**** Verify shape= {shape}")

    assert(len(shape)==2), f"Invalid shape < {len(shape)} > of weight matrix"
    if causal_mask_used:
        for i in range(shape[0]):   # shape[0] is T (no. of rows)
            min_wi = np.min(at_weights[i,0:i+1])
            del_wi = np.max(at_weights[i,0:i+1]) - min_wi
            at_weights[i,0:i+1] -= min_wi
            if del_wi != 0:
                at_weights[i,0:i+1] /= del_wi
    else:
        for i in range(shape[0]):   # shape[0] is T (no. of rows)
            min_wi = np.min(at_weights[i,0:])
            del_wi = np.max(at_weights[i,0:]) - min_wi
            at_weights[i,0:] -= min_wi
            if del_wi != 0:
                at_weights[i,0:] /= del_wi
            
    return at_weights

def viz_op_traj_with_attention(txy_preds_list,
                        all_at_mat_list, # could be enc_sa, dec_sa, dec_ga
                        path_lens,
                        mode='dec_sa',       #or 'a_s_attention'
                        layer_idx=0,
                        batch_idx=0, 
                        average_across_layers=False,
                        scale_each_row=True,
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        plot_flow=False,
                        at_time=None,
                        model_name=''
                        ):
    
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "att" + mode + model_name + "_@t"+str(at_time)+ ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.title(f"{mode}")
    # enc_sa_arr , dec_sa_arr, dec_ga_arr = all_att_mats
    # 0             1           2
    if mode == 'enc_sa':
        weights_list = [item[0] for item in all_at_mat_list]
    elif mode == 'dec_sa':
        weights_list = [item[1] for item in all_at_mat_list]
    elif mode == 'dec_ga':
        weights_list = [item[2] for item in all_at_mat_list]
    else:
        raise ValueError(f'no such mode :{mode}')
    
    

        
    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(txy_preds_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    for idx,traj in enumerate(txy_preds_list):
        if idx in traj_idx:
            states = txy_preds_list[idx]
            t_done =  path_lens[idx]
            weights = weights_list[idx]
            if average_across_layers:
                layer_str = 'avg'
                weights = np.mean(weights[:,batch_idx,:,:],axis=0)
            else:
                layer_str = str(layer_idx)
                weights = weights[layer_idx,batch_idx,:,:]
            if scale_each_row:
                scale_str = '_scaled_rows_'
                weights = scale_attention_rows(weights)
                
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done
            t_done = at_time

            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if stats!=None:
                mean, std = stats
                states = (states*std) + mean
        
            # Plot states
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            for t in range(t_done):
                plt.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                            c=cm.Reds(weights[t_done-1,t])) #TODO: verify weights idx


            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=0.5)

    # plot target area and set limits
    if env != None:
        setup_ax(ax,env)
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        # obstacle = DOLS_obstacle()
        # ax.add_patch(obstacle)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)

        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)

    plt.savefig(fname, bbox_inches="tight", dpi=600)
    wandbfname = "pred_trajs_with_" + mode
    if log_wandb:
        wandb.log({wandbfname: wandb.Image(fname)})

    plt.close()
    return fname

def setup_ax(ax,env, show_xlabel= True, 
                            show_ylabel=True, 
                            show_states=True,
                            show_xticks=True,
                            show_yticks=True,
                            lab_fs = 15,
                            tick_fs = 14,
                        ):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    xticks = np.arange(0,100+25,25)
    yticks = xticks.copy()
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)
    ax.set_xticklabels(xticks, fontsize=tick_fs)
    ax.set_yticklabels(yticks, fontsize=tick_fs)

    if not show_xticks:
        ax.tick_params(axis='x',       
                        which='both',      
                        bottom=False,      
                        labelbottom=False,
                        labelsize=tick_fs)
    if not show_yticks:
        ax.tick_params(axis='y',       
                which='both',      
                left=False,      
                labelleft=False,
                labelsize=tick_fs)
    xlabel = f"X "
    ylabel = f"Y "
    xlabel += "(Non-Dim)"
    ylabel += "(Non-Dim)"
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=lab_fs)
    if show_ylabel:
        ax.set_ylabel(ylabel,fontsize=lab_fs)
    if show_states:
        ax.scatter(env.start_pos[0], env.start_pos[1], color='k', marker='o')
    
        if env.target_pos.ndim == 1:
            ax.scatter(env.target_pos[0], env.target_pos[1], color='k', marker='*')
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                ax.scatter(target_pos[0], target_pos[1], color='k', marker='*')
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
                
                
                
                
def n_nans(var): 
    import numpy as np
    shape = var.shape
    return np.sum(np.int32(np.isnan(var.cpu().detach().numpy()))),   len(var.reshape(-1,))


"""
from my-decision-transformer/src_utils
"""
def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


class create_dt_dataset(Dataset):
    def __init__(self, dataset, idx_set, context_len, norm_params_4_val=None):
        """
        trajectories; 
        context_len:
        rtg_scale:
        state_dim: if = 5, then state is 'txyuv'
        
        dataset: [(Yi_r, obs_r, actions_r, states_r,
                    timesteps_r, dones_r, success_r, 
                    target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]
        states =  (t,x,y,xo,yo,wo)

        TODO: Shubham
        1. Change batch norm to layer norm
        2. states is just (txy)
        3. return env_embedding_seq (see pptt code for this)
        
        """
  
        self.context_len = context_len
        # TODO: Change if not pkl file
        self.dataset = dataset

        self.states = []
        for item in dataset:
            if len(item[3]) <= 120:
                self.states.append(np.concatenate([item[3],item[1][0:len(item[3])]],axis=-1).astype(np.float32))
            else:
                a = np.concatenate([item[1][0:120],item[1][0:len(item[3])-120]],axis=0)
                self.states.append(np.concatenate([item[3],a],axis=-1).astype(np.float32))
        # self.states = [item[3] for item in self.dataset]
        # try:
        #     self.states = [np.concatenate([item[3],item[1][0:len(item[3])]],axis=-1).astype(np.float32) for item in self.dataset]
        # except:
        #     self.states = [np.concatenate([item[3],item[1][0:120],item[1][0:len(item[3])-120]],axis=-1).astype(np.float32) for item in self.dataset]

        states = np.concatenate(self.states.copy(), axis=0).astype(np.float32)
        eps = 1e-8
        self.states_mean, self.states_std = np.mean(states, axis=0), np.std(states, axis=0) + eps

        # extract actions (tgt) and scale them to range [0,1)
        self.actions = [item[2]/(2*np.pi) for item in self.dataset]
        # normalise
        if norm_params_4_val == None:
            for i in range(len(self.states)):
                self.states[i] = self.states[i] - self.states_mean
                self.states[i] = np.divide(self.states[i], self.states_std + eps)

        else:
            tr_states_mean, tr_states_std= norm_params_4_val
            for i in range(len(self.states)):
                self.states[i] = self.states[i] - tr_states_mean
                self.states[i] = np.divide(self.states[i], tr_states_std + eps)  

        print("dummy")
        

    def get_src_stats(self):
        return self.states_mean, self.states_std

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        _, obs, _, _, _, _, _, target_pos, _, flow_dir, rzn = self.dataset[idx]
        states = self.states[idx]
        actions = self.actions[idx]
        traj_len = len(states)              #traj['states'].shape[0]
        
        if traj_len > self.context_len:
            # raise ValueError(" case not coded for")
            si = random.randint(0, traj_len - self.context_len)
            states = torch.from_numpy(states[si : si + self.context_len])
            actions = torch.from_numpy(actions[si : si + self.context_len])
            returns_to_go = torch.zeros((self.context_len, 1),
                                dtype=torch.float32)            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)        
        

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(states)

            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(actions)
            actions = torch.cat([actions,
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.zeros((self.context_len, 1),
                                dtype=torch.float32)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)
            
        target_pos = torch.from_numpy(np.array(target_pos))

        # dummy_t = float(int(0.8*self.context_len)) # time stamp for target state.
        # target_state = np.insert(target_state,0,dummy_t,axis=0)
        # return timesteps, states, actions, returns_to_go, traj_mask, target_state
        return timesteps, actions, traj_mask, target_pos, states, traj_len, idx, flow_dir, rzn, returns_to_go, obs


class create_dt_dataset_v2(Dataset):
    def __init__(self, dataset, idx_set, context_len, norm_params_4_val=None):
        """
        trajectories; 
        context_len:
        rtg_scale:
        state_dim: if = 5, then state is 'txyuv'
        
        dataset: [(Yi_r, obs_r, actions_r, states_r,
                    timesteps_r, dones_r, success_r, 
                    target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]
        
        states =  (ph1,ph2,phi3..,xo,yo,wo)
        """
  
        self.context_len = context_len
        # TODO: Change if not pkl file
        self.dataset = dataset
        self.states = [np.concatenate([item[0],item[1]],axis=-1).astype(np.float32) for item in self.dataset]


        # self.states = []
        # for item in dataset:
        #     if len(item[3]) <= 120:
        #         self.states.append(np.concatenate([item[0],item[1][0:len(item[3])]],axis=-1).astype(np.float32))
        #     else:
        #         a = np.concatenate([item[1][0:120],item[1][0:len(item[3])-120]],axis=0)
        #         self.states.append(np.concatenate([item[3],a],axis=-1).astype(np.float32))
        # self.states = [item[3] for item in self.dataset]
        # try:
        #     self.states = [np.concatenate([item[3],item[1][0:len(item[3])]],axis=-1).astype(np.float32) for item in self.dataset]
        # except:
        #     self.states = [np.concatenate([item[3],item[1][0:120],item[1][0:len(item[3])-120]],axis=-1).astype(np.float32) for item in self.dataset]

        states = np.concatenate(self.states.copy(), axis=0).astype(np.float32)
        eps = 1e-8
        self.states_mean, self.states_std = np.mean(states, axis=0), np.std(states, axis=0) + eps

        # extract actions (tgt) and scale them to range [0,1)
        self.actions = [item[2]/(2*np.pi) for item in self.dataset]
        # normalzise
        if norm_params_4_val == None:
            for i in range(len(self.states)):
                self.states[i] = self.states[i] - self.states_mean
                self.states[i] = np.divide(self.states[i], self.states_std + eps)

        else:
            tr_states_mean, tr_states_std= norm_params_4_val
            for i in range(len(self.states)):
                self.states[i] = self.states[i] - tr_states_mean
                self.states[i] = np.divide(self.states[i], tr_states_std + eps)  

        print("dummy")
        

    def get_src_stats(self):
        return self.states_mean, self.states_std

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        _, obs, _, states_p, _, _, _, target_pos, _, flow_dir, rzn = self.dataset[idx]
        states = self.states[idx]
        actions = self.actions[idx]
        traj_len = len(states_p)              #traj['states'].shape[0]
        
        if traj_len >= self.context_len:
            # raise ValueError(" case not coded for")
            si = random.randint(0, traj_len - self.context_len)
            states = torch.from_numpy(states[si : si + self.context_len])
            actions = torch.from_numpy(actions[si : si + self.context_len])
            returns_to_go = torch.zeros((self.context_len, 1),
                                dtype=torch.float32)            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)        
        

        else:
            padding_len = self.context_len - traj_len
            padding_len_4states = self.context_len - 120 
            # padding with zeros
            states = torch.from_numpy(states)

            states = torch.cat([states,
                                torch.zeros(([padding_len_4states] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(actions)
            actions = torch.cat([actions,
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.zeros((self.context_len, 1),
                                dtype=torch.float32)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)
            
        target_pos = torch.from_numpy(np.array(target_pos))

        # dummy_t = float(int(0.8*self.context_len)) # time stamp for target state.
        # target_state = np.insert(target_state,0,dummy_t,axis=0)
        # return timesteps, states, actions, returns_to_go, traj_mask, target_state
        return timesteps, actions, traj_mask, target_pos, states, traj_len, idx, flow_dir, rzn, returns_to_go, obs



class create_dt_dataset_v3(Dataset):
    def __init__(self, dataset, idx_set, context_len, norm_params_4_val=None, use_txyuv_as_states=False):
        """
        for train_dt_v3 
        - to compare performance against pptt on perlin
        - no obstacles
        - Pptt has yi info in encoder. Here we will not use Yi info
                
        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, rewards_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]
                <same as that given to create_envEnc_dtDec_dataset>
        states =  (t,x,y)
        """
  
  
        self.context_len = context_len
        # TODO: Change if not pkl file
        self.dataset = dataset
        self.use_txyuv_as_states = use_txyuv_as_states
        self.states = []
        if use_txyuv_as_states:
            state_access_id = -2
        else:
            state_access_id = 3
            
        for item in dataset:
            if len(item[state_access_id]) <= 61:
                self.states.append(item[state_access_id].astype(np.float32))
            else:
                print()
                # a = np.concatenate([item[1][0:120],item[1][0:len(item[3])-120]],axis=0)
                # self.states.append(np.concatenate([item[3],a],axis=-1).astype(np.float32))
                raise ValueError
                 
        states = np.concatenate(self.states.copy(), axis=0).astype(np.float32)
        eps = 1e-8
        self.states_mean, self.states_std = np.mean(states, axis=0), np.std(states, axis=0) + eps

        # extract actions (tgt) and scale them to range [0,1)
        self.actions = [item[2]/(2*np.pi) for item in self.dataset]
        # normalise
        if norm_params_4_val == None:
            for i in range(len(self.states)):
                self.states[i] = self.states[i] - self.states_mean
                self.states[i] = np.divide(self.states[i], self.states_std + eps)

        else:
            tr_states_mean, tr_states_std= norm_params_4_val
            for i in range(len(self.states)):
                self.states[i] = self.states[i] - tr_states_mean
                self.states[i] = np.divide(self.states[i], tr_states_std + eps)          

    def get_src_stats(self):
        return self.states_mean, self.states_std

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        Yi_r, _, _, _,  timesteps, rews, dones, success, target_pos, st_pos, flow_dir, rzn, _, v_  = self.dataset[idx]
        # Yi_r, _, _, _,  timesteps, rews, dones, success, target_pos, st_pos, flow_dir, rzn  = self.dataset[idx]
              
        states = self.states[idx]
        actions = self.actions[idx]
        traj_len = len(states)              #traj['states'].shape[0]
        
        if traj_len > self.context_len:
            raise ValueError(" case not coded for")
            # copy old logic from original crate_dt_datset() if needed
        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(states)

            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(actions)
            actions = torch.cat([actions,
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.zeros((self.context_len, 1),
                                dtype=torch.float32)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)
            
        target_pos = torch.from_numpy(np.array(target_pos))

        return timesteps, actions, traj_mask, target_pos, states, traj_len, idx, flow_dir, rzn, returns_to_go

    
def plot_grad_flow(named_parameters, title_suffix=""):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.cla()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001) # zoom in on the lower gradient regions
    # log scale in y axis 
    # plt.yscale('log')
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow"+title_suffix)
    plt.grid(True)
    plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tmp/grads.png')