import sys
import numpy as np

from root_path import ROOT
from os.path import join

sys.path.insert(0, ROOT)
import seaborn as sns
import wandb
from datetime import datetime
import imageio.v2 as imageio
from src_utils import setup_env
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from src_utils import scale_attention_rows
import matplotlib.patches as patches
import torch


# TODO: remove all hardcode!!
class paper_plots:
    def __init__(self, env,  op_traj_dict_list, stats, paper_plot_info, non_dim_plots=True, save_dir='../tmp/'):
        self.env = env
        self.op_traj_dict_list = op_traj_dict_list
        self.stats = stats  #training mean and variance for normalization
        self.paper_plot_info = paper_plot_info
        self.save_dir = save_dir
        self.non_dim_plots = non_dim_plots

    def plot_all(self):
        self.plot_traj_by_arr()
        self.plot_traj_att()
        self.plot_att_heatmap()


    def plot_vel_field(self,ax, obs_mask, t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
        # Make modes the last axis
        Ui = np.transpose(self.env.Ui,(0,2,3,1))
        Vi = np.transpose(self.env.Vi,(0,2,3,1))
        vx_grid = self.env.U[t,:,:] + np.dot(Ui[t,:,:,:],self.env.Yi[t,r,:])
        vy_grid = self.env.V[t,:,:] + np.dot(Vi[t,:,:,:],self.env.Yi[t,r,:])
        vx_grid = np.flipud(vx_grid)
        vy_grid = np.flipud(vy_grid)
        Xs = np.arange(0,self.env.xlim) + (self.env.dxy/2)
        Ys = np.arange(0,self.env.ylim) + (self.env.dxy/2)
        X,Y = np.meshgrid(Xs, Ys)
        ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
        v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
        v_mag_grid[np.where(obs_mask==1)]=np.nan
        ax.contourf(X, Y, v_mag_grid, alpha = 0.9, zorder = -1e5, colors='white')
        ax.set_facecolor("grey") 
        im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)
        # ax = plt.gca()
        # ax.set_facecolor('grey')
        return im
        
    def plot_vel_field_perlin(self,ax, obs_mask, t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
        # Make modes the last axis
        # Ui = np.transpose(self.env.Ui,(0,2,3,1))
        # Vi = np.transpose(self.env.Vi,(0,2,3,1))
        vx_grid = self.env.U[t,:,:] #+ np.dot(Ui[t,:,:,:],self.env.Yi[t,r,:])
        vy_grid = self.env.V[t,:,:] #+ np.dot(Vi[t,:,:,:],self.env.Yi[t,r,:])
        vx_grid = np.flipud(vx_grid)
        vy_grid = np.flipud(vy_grid)
        Xs = np.arange(0,self.env.xlim) + (self.env.dxy/2)
        Ys = np.arange(0,self.env.ylim) + (self.env.dxy/2)
        X,Y = np.meshgrid(Xs, Ys)
        ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
        v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
        v_mag_grid[np.where(obs_mask==1)]=np.nan
        ax.contourf(X, Y, v_mag_grid, alpha = 0.9, zorder = -1e5, colors='white')
        ax.set_facecolor("grey") 
        im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)
        # ax = plt.gca()
        # ax.set_facecolor('grey')
        return im

    def plot_traj_by_arr(self, traj_dataset, set_str=""):
        info = self.paper_plot_info["trajs_by_arr"]
        t_dones = []
        for traj in traj_dataset:
            _,_,_,_, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # vmax = 51
        # print(f"---- Info: tdone: min={vmin}, max={vmax} ")
        plt.hist(t_dones)
        plt.savefig("/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/tdone-hist")
        fig, ax = plt.subplots()

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax, r=499)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask, _ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
           
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # if idx>10:
            #     break


        cbar_fontsize = 12
        cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1, 3)])
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
        
        cbarv = fig.colorbar(im, ax=ax)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)
        fname = info["fname"] + set_str
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        return ax
    
        
    def plot_val_ip_op(self, traj_dataset, obs_mask,
                       preds_list,
                       path_lens,
                       success_list,
                        at_time=None):
        """
        Plots the input and output trajectories for the given dataset.

        Parameters:
            traj_dataset (list): The dataset containing trajectory information.
            obs_mask (list): The mask for observed values.
            preds_list (list): The list of predicted trajectories.
            path_lens (list): The lengths of the trajectories.
            success_list (list): The list of success values.
            at_time (int, optional): The time at which to plot the trajectories. Defaults to None.

        Returns:
            None
        """
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))

        info = self.paper_plot_info["plot_val_ip_op"]
        
        ip_states_list =[item[3] for item in traj_dataset.dataset] # scale in 100
        ip_path_lens = [len(item[2]) for item in traj_dataset.dataset] #item[2]
        # vmin = min(path_lens + ip_path_lens)
        # vmax = max(path_lens + ip_path_lens)
        vmin = min(ip_path_lens)
        vmax = max(ip_path_lens)

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)


        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field_perlin(ax, obs_mask, t=59,r=499)
   
        
        for idx,traj in enumerate(ip_states_list):
            states = ip_states_list[idx]
            t_done = ip_path_lens[idx]
        #     pr_t_dones.append(t_done)
        #     # Plot sstates
        #     # shape: (eval_batch_size, max_test_ep_len, state_dim)
        #     if t_done < 68:
            # ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done))

        #         # ax.scatter(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done),s=1)
        pr_t_dones = []
        ax = axs[1]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field_perlin(ax,obs_mask,t=59,r=499)

        for idx, traj in enumerate(preds_list):
            states = preds_list[idx]
            t_done = path_lens[idx] 
            # if success_list[idx]:
                # ax.scatter(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1, s=1 )
            ax.plot(states[0,:t_done,1], states[0,:t_done,2], color=sm.to_rgba(t_done), alpha=0.5 )
                # ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        summary = {}
        summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
        summary["std Tarr logged dataset"] = np.std(ip_path_lens)
        summary["mean Tarr prediction" ] = np.mean(path_lens)
        summary["std Tarr prediction" ] = np.std(path_lens)
        summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
        summary["prediction count"] = len(success_list)
        print("------ SUMMARY-------\n", summary)
        # cbar_fontsize = 12
        # cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1)])
        # cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
        
        # cbarv = fig.colorbar(im, ax=ax)
        # cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 15
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)


        fname = info["fname"] 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)

    def plot_val_ip_op_target_specific(self, traj_dataset, obs_mask,target_id=1,target_denorm=np.array([40., 40.]),at_time=None,success=True):
        """
        TODO: Have to edit (Shubham)
        Plots the input and output trajectories for the given dataset.

        Parameters:
            traj_dataset (list): The dataset containing trajectory information.
            obs_mask (list): The mask for observed values.
            preds_list (list): The list of predicted trajectories.
            path_lens (list): The lengths of the trajectories.
            success_list (list): The list of success values.
            at_time (int, optional): The time at which to plot the trajectories. Defaults to None.

        Returns:
            None
        """
        op_traj_dict_list = [item for item in self.op_traj_dict_list if item['target_id']==target_id]
        # op_traj_dict_list = [op_traj_dict_list[0]]

        # fail_list = [item for item in op_traj_dict_list if item['success']==False]
        # suc_list = [item for item in op_traj_dict_list if item['success']==True]
        
        states_mean, states_std = self.stats
        preds_list = [d['states']*states_std + states_mean for d in op_traj_dict_list]
        # mask = ~(tensor[:, :, 0] == 1.6226e+01) & ~(tensor[:, :, 1] == 2.1727e+01) & ~(tensor[:, :, 2] == 2.2559e+01)
        
        path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
        success_list = [d['success'] for d in op_traj_dict_list]
        
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))

        # target_denorm = (target*self.stats[1][1:]) + self.stats[0][1:]
        # target_denorm = torch.round(target_denorm).numpy()
        self.env.target_pos = target_denorm
        info = self.paper_plot_info["plot_val_ip_op_target_specific"]
        
        flow_dir_to_index = {item['flow_dir']: index for index, item in enumerate(op_traj_dict_list)}
                
        # ip_states_list = [item[3] for item in traj_dataset.dataset if (isinstance(item[8], np.ndarray) and np.array_equal(item[8], target_denorm))]
        # ip_path_lens = [len(item[2]) for item in traj_dataset.dataset if (isinstance(item[8], np.ndarray) and np.array_equal(item[8], target_denorm))]
        ip_states_list = [item[3] 
                        for item in traj_dataset.dataset 
                        if item[-2] in flow_dir_to_index 
                        and op_traj_dict_list[flow_dir_to_index[item[-2]]]['flow_dir'] == item[-2]
                        and isinstance(item[8], np.ndarray) 
                        and np.array_equal(item[8], target_denorm)
                    ]
        ip_path_lens = [len(item[2]) 
                        for item in traj_dataset.dataset 
                        if item[-2] in flow_dir_to_index 
                        and op_traj_dict_list[flow_dir_to_index[item[-2]]]['flow_dir'] == item[-2]
                        and isinstance(item[8], np.ndarray) 
                        and np.array_equal(item[8], target_denorm)
                    ]
        # vmin = min(path_lens + ip_path_lens)
        # vmax = max(path_lens + ip_path_lens)
        vmin = min(ip_path_lens)
        vmax = max(ip_path_lens)

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)


        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field_perlin(ax, obs_mask, t=59,r=499)
   
        
        for idx,traj in enumerate(ip_states_list):
            states = ip_states_list[idx]
            t_done = ip_path_lens[idx]
            ax.plot(states[:t_done+1,1], states[:t_done+1,2], color=sm.to_rgba(t_done))

        pr_t_dones = []
        ax = axs[1]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field_perlin(ax,obs_mask,t=59,r=499)

        for idx, traj in enumerate(preds_list):
            states = preds_list[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx]
            if success: 
                if success_list[idx]:
                    ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
            else:
                if not success_list[idx]:
                    ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))

        summary = {}
        summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
        summary["std Tarr logged dataset"] = np.std(ip_path_lens)
        summary["mean Tarr prediction" ] = np.mean(path_lens)
        summary["std Tarr prediction" ] = np.std(path_lens)
        summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
        summary["prediction count"] = len(success_list)
        print(f"------ SUMMARY_{target_denorm}_Success={success}-------\n", summary)
        
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 15
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)

        plt.suptitle(f'Target: {target_denorm}', fontsize=15)
        fname = info["fname"]+f"_{target_id}_Success={success}"
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        

        
    def plot_success_non_success(self, traj_dataset, obs_mask):
        self.plot_val_ip_op_target_specific(traj_dataset, obs_mask,target_id=0,target_denorm=np.array([40., 40.]), success=True)
        # self.plot_val_ip_op_target_specific(traj_dataset, obs_mask,target_id=1,target_denorm=np.array([35., 25.]), success=True)
        # self.plot_val_ip_op_target_specific(traj_dataset, obs_mask,target_id=2,target_denorm=np.array([25., 35.]), success=True)
        self.plot_val_ip_op_target_specific(traj_dataset, obs_mask,target_id=0,target_denorm=np.array([40., 40.]), success=False)
        # self.plot_val_ip_op_target_specific(traj_dataset, obs_mask,target_id=1,target_denorm=np.array([35., 25.]), success=False)
        # self.plot_val_ip_op_target_specific(traj_dataset, obs_mask,target_id=2,target_denorm=np.array([25., 35.]), success=False)

    def plot_specific_perlin_path(self, traj_dataset, obs_mask,target_id=1,target_denorm=np.array([40., 40.]),at_time=None,success=True):
        """
        TODO: Have to edit (Shubham)
        Plots the input and output trajectories for the given dataset for specific perlin flow 

        Parameters:
            traj_dataset (list): The dataset containing trajectory information.
            obs_mask (list): The mask for observed values.
            preds_list (list): The list of predicted trajectories.
            path_lens (list): The lengths of the trajectories.
            success_list (list): The list of success values.
            at_time (int, optional): The time at which to plot the trajectories. Defaults to None.

        Returns:
            None
        """
        op_traj_dict_list = [item for item in self.op_traj_dict_list if item['target_id']==target_id and item['success']==True]
        op_traj_dict_list = [op_traj_dict_list[3]]
        flow_dir = op_traj_dict_list[0]['flow_dir']
        # fail_list = [item for item in op_traj_dict_list if item['success']==False]
        # suc_list = [item for item in op_traj_dict_list if item['success']==True]
        self.env = setup_env(flow_dir, target_id=-1)
        states_mean, states_std = self.stats
        preds_list = [d['states']*states_std + states_mean for d in op_traj_dict_list]
        # mask = ~(tensor[:, :, 0] == 1.6226e+01) & ~(tensor[:, :, 1] == 2.1727e+01) & ~(tensor[:, :, 2] == 2.2559e+01)
        
        path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
        success_list = [d['success'] for d in op_traj_dict_list]
        
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))

        # target_denorm = (target*self.stats[1][1:]) + self.stats[0][1:]
        # target_denorm = torch.round(target_denorm).numpy()
        self.env.target_pos = target_denorm
        info = self.paper_plot_info["plot_specific_perlin_path"]
        
        flow_dir_to_index = {item['flow_dir']: index for index, item in enumerate(op_traj_dict_list)}
                
        # ip_states_list = [item[3] for item in traj_dataset.dataset if (isinstance(item[8], np.ndarray) and np.array_equal(item[8], target_denorm))]
        # ip_path_lens = [len(item[2]) for item in traj_dataset.dataset if (isinstance(item[8], np.ndarray) and np.array_equal(item[8], target_denorm))]
        ip_states_list = [item[3] 
                        for item in traj_dataset.dataset 
                        if item[-2] in flow_dir_to_index 
                        and op_traj_dict_list[flow_dir_to_index[item[-2]]]['flow_dir'] == item[-2]
                        and isinstance(item[8], np.ndarray) 
                        and np.array_equal(item[8], target_denorm)
                    ]
        ip_path_lens = [len(item[2]) 
                        for item in traj_dataset.dataset 
                        if item[-2] in flow_dir_to_index 
                        and op_traj_dict_list[flow_dir_to_index[item[-2]]]['flow_dir'] == item[-2]
                        and isinstance(item[8], np.ndarray) 
                        and np.array_equal(item[8], target_denorm)
                    ]
        # vmin = min(path_lens + ip_path_lens)
        # vmax = max(path_lens + ip_path_lens)
        vmin = min(ip_path_lens)
        vmax = max(ip_path_lens)

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)


        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field_perlin(ax, obs_mask, t=59,r=499)
   
        
        for idx,traj in enumerate(ip_states_list):
            states = ip_states_list[idx]
            t_done = ip_path_lens[idx]
            ax.plot(states[:t_done+1,1], states[:t_done+1,2], color=sm.to_rgba(t_done))

        pr_t_dones = []
        ax = axs[1]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field_perlin(ax,obs_mask,t=59,r=499)

        for idx, traj in enumerate(preds_list):
            states = preds_list[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx]
            if success: 
                if success_list[idx]:
                    ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
            else:
                if not success_list[idx]:
                    ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))

        summary = {}
        summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
        summary["std Tarr logged dataset"] = np.std(ip_path_lens)
        summary["mean Tarr prediction" ] = np.mean(path_lens)
        summary["std Tarr prediction" ] = np.std(path_lens)
        summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
        summary["prediction count"] = len(success_list)
        print(f"------ SUMMARY_{target_denorm}_Success={success}-------\n", summary)
        
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 15
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)

        flow_dir = np.int32(flow_dir.split('/')[-1].split('_')[-1])
        plt.suptitle(f'Target: {target_denorm} Flow_dir: {flow_dir}', fontsize=15)
        fname = info["fname"]+f"_{target_id}_Success={success}_Flow_dir={flow_dir}.png"
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600) 
    
                
    def plot_actions(self, traj_dataset,
                       preds_list,
                       path_lens,
                       success_list,
                       actions,
                        at_time=None):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))

        info = self.paper_plot_info["plot_actions"]
        
        ip_actions_list = [item[1] for item in traj_dataset]
        vmax = int(78)
        ax = axs[0]
        # self.setup_ax(ax)       
        # im = self.plot_vel_field(ax,t=vmax,r=9999)
        # obstacle = self.DOLS_obstacle()
        # ax.add_patch(obstacle)
        
        for i in range(len(ip_actions_list)):
            ax.plot(torch.squeeze(ip_actions_list[i]))
 

        ax = axs[1] 
        # self.setup_ax(ax,show_ylabel=False)
        # im = self.plot_vel_field(ax,t=vmax,r=9999)
        # # self.plot_obstacle(ax, xyw=xyw)
        # obstacle = self.DOLS_obstacle()
        # ax.add_patch(obstacle)

        for i in range(len(actions)):
            ax.plot((torch.squeeze(actions[i][0])))
        
        fname = info["fname"] 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)

    def plot_train_val_ip_op(self, tr_traj_dataset, val_traj_dataset,
                            at_time=None):
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15,5))

        info = self.paper_plot_info["plot_train_val_ip_op"]
        t_dones = []
        for traj in val_traj_dataset:
            _,_,_,_, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones,119)

        # vmax = 51

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax,r=9999)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(tr_traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=0.2 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        ax = axs[1]
        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax,r=9999)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(val_traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        pr_t_dones = []
        ax = axs[2]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field(ax,t=vmax,r=9999)
        print(f"{len(self.op_traj_dict_list)}")
        # sys.exit()
        for idx,traj in enumerate(self.op_traj_dict_list):
            states = traj['states']
            t_done =  traj['t_done']
            pr_t_dones.append(t_done)
            mean, std = self.stats
            states = (states*std) + mean
        
            # Plot sstates
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
                
        wandb.run.summary["mean Tarr logged dataset"] = np.mean(t_dones)
        wandb.run.summary["std Tarr logged dataset"] = np.std(t_dones)
        wandb.run.summary["mean Tarr prediction" ] = np.mean(pr_t_dones)
        wandb.run.summary["std Tarr prediction" ] = np.std(pr_t_dones)
     
        # cbar_fontsize = 12
        # cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1)])
        # cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
        
        # cbarv = fig.colorbar(im, ax=ax)
        # cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 14
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)


        fname = info["fname"] 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)





    def plot_att_trajs_at_t(self,ax,at_time,mode,stats):
        for idx,traj in enumerate(self.op_traj_dict_list):
            states = traj['states']
            t_done =  traj['t_done']
            # print(f"-------- verify : states= {states}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done
           
            # Rescale
            mean, std = stats
            states = (states*std) + mean

            at_weights = traj['attention_weights'][0,0,:,:].cpu().detach().numpy()
            a_s_wts_scaled = scale_attention_rows(at_weights[2::3,1::3])
            a_a_wts_scaled = scale_attention_rows(at_weights[2::3,2::3])

            alpha = 0.7
            ax.scatter(states[0,at_time,1], states[0,at_time,2], c='k', marker='p')
            if mode == 'a_a_attention':
                for t in range(t_done):
                    im =ax.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Reds(a_a_wts_scaled[t_done-1,t]), zorder=10, alpha=alpha)
            elif mode == 'a_s_attention':
                for t in range(t_done):
                    im = ax.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Greens(a_s_wts_scaled[t_done-1,t]), zorder=10, alpha=alpha)
            else:
                raise Exception("invalid argument for mode")
        return im

    def plot_traj_by_att(self, mode):
        if mode == 'a_a_attention':
            cmap = cm.Reds
        elif mode == 'a_s_attention':
            cmap = cm.Greens
        else:
            raise Exception("invalid argument for mode")
        info = self.paper_plot_info["trajs_by_att"]
        nplots = len(info["ts"])
       
        fig, axs = plt.subplots(1,nplots, sharey=True, figsize=(15,5))
        # fig.suptitle('')    #title of overall plot
        show_ylabel = True
        for i in range(nplots):
            if i>=1:
                show_ylabel=False
            ax = axs[i]
            at_time = info["ts"][i]
            self.setup_ax(ax,show_ylabel=show_ylabel)
            self.plot_att_trajs_at_t(ax, at_time, mode, self.stats)
            im = self.plot_vel_field(ax,at_time,g_strmplot_lw=0.5, g_strmplot_arrowsize=0.5)
            ax.set_title(f"t={at_time}")
        
        # colorbars
        # cbar1_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_fontsize = 12
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Attention Weights (scaled)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.65)
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        fname = info["fname"] + "_" + mode
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        return


    def setup_ax(self, ax, show_xlabel= True, 
                            show_ylabel=True, 
                            show_states=True,
                            show_xticks=True,
                            show_yticks=True,
                            lab_fs = 15,
                            tick_fs = 14,
                            ):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0,self.env.xlim])
        ax.set_ylim([0,self.env.ylim])
        xticks = np.arange(0,self.env.xlim +25,25)
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
        if self.non_dim_plots == True:
            xlabel += "(Non-Dim)"
            ylabel += "(Non-Dim)"
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=lab_fs)
        if show_ylabel:
            ax.set_ylabel(ylabel,fontsize=lab_fs)
        if show_states:
            ax.scatter(self.env.start_pos[0], self.env.start_pos[1], color='k', marker='o')
        
            if self.env.target_pos.ndim == 1:
                ax.scatter(self.env.target_pos[0], self.env.target_pos[1], color='k', marker='*')
                target_circle = plt.Circle(self.env.target_pos, self.env.target_rad, color='r', alpha=0.5)
                ax.add_patch(target_circle)
            elif self.env.target_pos.ndim > 1:
                for target_pos in self.env.target_pos:
                    ax.scatter(target_pos[0], target_pos[1], color='k', marker='*')
                    target_circle = plt.Circle(target_pos, self.env.target_rad, color='r', alpha=0.5)
                    ax.add_patch(target_circle)


    # TODO: Change as per causal_mask
    def plot_att_heatmap(self, set_idx=0, sample_idx=0):
        """
        attention_weights: weight matrix expected shape = 1(or 64),1,210,210 or B,N,T,T where T is 3*context_len
        idx: sample index of batch
        scale_each_row: scales each row INDEPENDENTLY to lie between 0 and 1 for visualization
        """
        info = self.paper_plot_info["att_heatmap"]
        op_traj_dict = self.op_traj_dict_list[set_idx]
        # attention weigghts in the last block
        attention_weights = op_traj_dict['attention_weights']
        # normalized_weights = F.softmax(attention_weights, dim=-1)

        weights = attention_weights.cpu().detach().numpy()
        shape = weights.shape
        
        # Plot attenetion scores for the ith trajectory/sample among the batch (for training batch)
        weights = weights[sample_idx,0,:,:]
        
        # scale each row visualization
        weights = scale_attention_rows(weights)

        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_ticks(np.arange(0,180,25))
        ax.yaxis.set_ticks(np.arange(0,180, 25))
        shw = ax.imshow(weights, cmap=cm.Reds)
        cbar = plt.colorbar(shw)
        cbar_fontsize = 12
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        # cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Attention Weights (scaled)", fontsize=cbar_fontsize)

        # title =  scale_str + info_string + "setIdx-" +  str(set_idx)
        # plt.title(title)
        # plt.figure(figsize=(12,10))
        # ax = sns.heatmap(weights, linewidth=0.05)
        fname = info["fname"]
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches="tight", dpi=600)

        return



    def load_velocity(self, vel_fname):
        U = np.load(join(vel_fname,"all_u_mat.npy"))
        V = np.load(join(vel_fname,"all_v_mat.npy"))
        Ui = np.load(join(vel_fname,"all_ui_mat.npy"))
        Vi = np.load(join(vel_fname,"all_vi_mat.npy"))
        Yi = np.load(join(vel_fname,"all_Yi.npy"))

        # replace nans with 0s and scale velocity as per vmax_by_F factor
        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        Ui[np.isnan(Ui)] = 0
        Vi[np.isnan(Vi)] = 0
        Yi[np.isnan(Yi)] = 0

        return (U, V, Ui, Vi, Yi)
    

    # def plot_val_ip_op(self, traj_dataset,
    #                    preds_list,
    #                    path_lens,
    #                    success_list,
    #                     at_time=None):
    #     fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))

    #     info = self.paper_plot_info["plot_val_ip_op"]
        
    #     ip_states_list =[item[3] for item in traj_dataset.dataset]
    #     ip_path_lens = [len(item[2]) for item in traj_dataset.dataset] #item[2]
    #     vmin = min(path_lens + ip_path_lens)
    #     vmax = max(path_lens + ip_path_lens)
    #     vmax = min(vmax, 119)
    #     vmax = int(78)
    #     vmin = int(72)
    #     # # vmax = 51

    #     # Make a user-defined colormap.
    #     cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    #     cmap = plt.get_cmap('YlOrRd')
    #     sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
    #     # Works for static obstacle only
    #     obs_token = traj_dataset.dataset[0][1][0] #sample, obstacle key, timestep
    #     xyw=(-10,-10,5)

    #     ax = axs[0]
    #     self.setup_ax(ax)       
    #     im = self.plot_vel_field(ax,t=vmax,r=1999)
    #     self.plot_obstacle(ax, xyw=xyw)
    #     # obstacle = self.DOLS_obstacle()
    #     # ax.add_patch(obstacle)


    #     for idx,traj in enumerate(ip_states_list):
    #         states = ip_states_list[idx]
    #         t_done = ip_path_lens[idx]
    #     #     pr_t_dones.append(t_done)
    #     #     # Plot sstates
    #     #     # shape: (eval_batch_size, max_test_ep_len, state_dim)
    #     #     if t_done < 68:
    #         # ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
    #         ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=0.5)

    #     #         # ax.scatter(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done),s=1)
    #     pr_t_dones = []
    #     ax = axs[1] 
    #     self.setup_ax(ax,show_ylabel=False)
    #     im = self.plot_vel_field(ax,t=vmax,r=1999)
    #     # self.plot_obstacle(ax, xyw=xyw)
    #     obstacle = self.DOLS_obstacle()
    #     ax.add_patch(obstacle)

    #     for idx, traj in enumerate(preds_list[:1500]):
    #         states = preds_list[idx]
    #         t_done = path_lens[idx] 
    #         if success_list[idx]:
    #             # ax.scatter(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1, s=1 )
    #             ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done), alpha=0.5 )
    #             # ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

    #     summary = {}
    #     summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
    #     summary["std Tarr logged dataset"] = np.std(ip_path_lens)
    #     summary["mean Tarr prediction" ] = np.mean(path_lens)
    #     summary["std Tarr prediction" ] = np.std(path_lens)
    #     summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
    #     summary["prediction count"] = len(success_list)
    #     print("------ SUMMARY-------\n", summary)


    #     plt.subplots_adjust(left= 0.1, right=0.9, top=1.1, bottom=0.4, wspace=0.1, hspace=0.3)
 
    #     cax_arr = ax.inset_axes([1.05, -0.20, 0.05, 1.2])
    #     cax_vel = ax.inset_axes([1.25, -0.20, 0.05, 1.2])
    #     cbar_fontsize = 16

    #     cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
    #     cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
    #     cbar.ax.tick_params(labelsize=13)
     
    #     cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
    #     cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)
    #     cbarv.ax.tick_params(labelsize=13)


    #     fname = info["fname"] 
    #     save_name = join(self.save_dir,fname)
    #     # plt.figure(layout='constrained')
    #     plt.savefig(save_name, bbox_inches='tight', dpi=600)
        

    # def plot_trajs_ip_op(self, traj_dataset,
    #                    preds_list,
    #                    path_lens,
    #                    success_list,
    #                     at_time=None):
    #     fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))

    #     info = self.paper_plot_info["plot_trajs_ip_op"]
        
    #     ip_states_list =[item[3] for item in traj_dataset.dataset]
    #     ip_path_lens = [len(item[2]) for item in traj_dataset.dataset] #item[2]
    #     # vmin = min(path_lens + ip_path_lens)
    #     # vmax = max(path_lens + ip_path_lens)
    #     # vmax = min(vmax, 119)
    #     vmax = int(78)
    #     vmin = int(72)


    #     ax = axs[0]
    #     self.setup_ax(ax)       
    #     im = self.plot_vel_field(ax,t=vmax,r=1999)
    #     obstacle = self.DOLS_obstacle()
    #     ax.add_patch(obstacle)
    #     # traj_dataset=random.shuffle(traj_dataset)
    #     color_dict = {}
    #     for idx,traj in enumerate(ip_states_list):
    #         states = ip_states_list[idx]
    #         t_done = ip_path_lens[idx]
    #             # if  (states[:,2][0][2] > 45):
    #         if  (states[1][2] > 45):    
    #             ax.plot(states[:t_done,1], states[:t_done,2], color='Green', alpha=1)
    #             color_dict[str(idx)] = 'Green'
    #         else:
    #             ax.plot(states[:t_done,1], states[:t_done,2], color='Yellow', alpha=1)
    #             color_dict[str(idx)] = 'Yellow'

    #     ax = axs[1] 
    #     self.setup_ax(ax,show_ylabel=False)
    #     im = self.plot_vel_field(ax,t=vmax,r=1999)
    #     # self.plot_obstacle(ax, xyw=xyw)
    #     obstacle = self.DOLS_obstacle()
    #     ax.add_patch(obstacle)

    #     for idx, traj in enumerate(preds_list):
    #         states = preds_list[idx]
    #         t_done = path_lens[idx] 
    #         if (success_list[idx]):
                
    #             ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=color_dict[str(idx)], alpha=1 )
                

    #     summary = {}
    #     summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
    #     summary["std Tarr logged dataset"] = np.std(ip_path_lens)
    #     summary["mean Tarr prediction" ] = np.mean(path_lens)
    #     summary["std Tarr prediction" ] = np.std(path_lens)
    #     summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
    #     summary["prediction count"] = len(success_list)
    #     print("------ SUMMARY-------\n", summary)

        
    #     plt.subplots_adjust(left= 0.1, right=0.9, top=1.1, bottom=0.4, wspace=0.1, hspace=0.3)

    #     cax_vel = ax.inset_axes([1.05, -0.20, 0.05, 1.2])

    #     cbar_fontsize = 16
     
    #     cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
    #     cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)
    #     cbarv.ax.tick_params(labelsize=13)

    #     fname = info["fname"] 
    #     save_name = join(self.save_dir,fname)
    #     plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
    
    
    
    
    def plot_val_ip_op_target_specific_not_successful(self, traj_dataset, obs_mask,target_id=1,target_denorm=np.array([40., 40.]),at_time=None,):
        """
        TODO: Have to edit (Shubham)
        Plots the input and output trajectories for the given dataset.

        Parameters:
            traj_dataset (list): The dataset containing trajectory information.
            obs_mask (list): The mask for observed values.
            preds_list (list): The list of predicted trajectories.
            path_lens (list): The lengths of the trajectories.
            success_list (list): The list of success values.
            at_time (int, optional): The time at which to plot the trajectories. Defaults to None.

        Returns:
            None
        """
        op_traj_dict_list = [item for item in self.op_traj_dict_list if item['target_id']==target_id]
        # op_traj_dict_list = [item for item in self.op_traj_dict_list if item['success']==False and item['target_id']==target_id]
        # op_traj_dict_list = [op_traj_dict_list[0]]
        # op_traj_dict_list_non_successful = [item for item in self.op_traj_dict_list if item['success']==False]
        # op_traj_dict_list = [item for item in self.op_traj_dict_list if item['success']==False and item['target_id']==target_id and item["flow_dir"]=='/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/Perlin/Mag_0.8/Perlin_g50x50x60_212']
        states_mean, states_std = self.stats
        preds_list = [d['states']*states_std + states_mean for d in op_traj_dict_list]
        # mask = ~(tensor[:, :, 0] == 1.6226e+01) & ~(tensor[:, :, 1] == 2.1727e+01) & ~(tensor[:, :, 2] == 2.2559e+01)
        
        path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
        success_list = [d['success'] for d in op_traj_dict_list]
        
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))

        # target_denorm = (target*self.stats[1][1:]) + self.stats[0][1:]
        # target_denorm = torch.round(target_denorm).numpy()
        self.env.target_pos = target_denorm
        info = self.paper_plot_info["plot_val_ip_op_target_specific_not_successful"]
        
        flow_dir_to_index = {item['flow_dir']: index for index, item in enumerate(op_traj_dict_list)}
                
        # ip_states_list = [item[3] for item in traj_dataset.dataset if (isinstance(item[8], np.ndarray) and np.array_equal(item[8], target_denorm))]
        # ip_path_lens = [len(item[2]) for item in traj_dataset.dataset if (isinstance(item[8], np.ndarray) and np.array_equal(item[8], target_denorm))]
        ip_states_list = [item[3] 
                        for item in traj_dataset.dataset 
                        if item[-2] in flow_dir_to_index 
                        and op_traj_dict_list[flow_dir_to_index[item[-2]]]['flow_dir'] == item[-2]
                        and isinstance(item[8], np.ndarray) 
                        and np.array_equal(item[8], target_denorm)
                    ]
        ip_path_lens = [len(item[2]) 
                        for item in traj_dataset.dataset 
                        if item[-2] in flow_dir_to_index 
                        and op_traj_dict_list[flow_dir_to_index[item[-2]]]['flow_dir'] == item[-2]
                        and isinstance(item[8], np.ndarray) 
                        and np.array_equal(item[8], target_denorm)
                    ]
        # vmin = min(path_lens + ip_path_lens)
        # vmax = max(path_lens + ip_path_lens)
        vmin = min(ip_path_lens)
        vmax = max(ip_path_lens)

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)


        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field_perlin(ax, obs_mask, t=59,r=499)
   
        
        for idx,traj in enumerate(ip_states_list):
            states = ip_states_list[idx]
            t_done = ip_path_lens[idx]
            ax.plot(states[:t_done+1,1], states[:t_done+1,2], color=sm.to_rgba(t_done))

        pr_t_dones = []
        ax = axs[1]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field_perlin(ax,obs_mask,t=59,r=499)

        for idx, traj in enumerate(preds_list):
            states = preds_list[idx]
            states = torch.where(states == states[-1, -1], torch.zeros_like(states), states)
            states = (states[(states != 0).any(dim=-1)]).unsqueeze(dim=0)
            t_done = path_lens[idx] 
            if not success_list[idx]:
                ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))

        summary = {}
        summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
        summary["std Tarr logged dataset"] = np.std(ip_path_lens)
        summary["mean Tarr prediction" ] = np.mean(path_lens)
        summary["std Tarr prediction" ] = np.std(path_lens)
        summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
        summary["prediction count"] = len(success_list)
        print(f"------ SUMMARY_{target_denorm}-------\n", summary)
        
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 15
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)

        plt.suptitle(f'Target: {target_denorm}', fontsize=15)
        fname = info["fname"]+f"_{target_id}"
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)