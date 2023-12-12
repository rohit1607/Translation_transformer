
import pickle
import torch
import numpy as np
import random
import time
import math
import yaml
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os
import collections
import matplotlib


def execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} exec time: {execution_time:.3f} secs")
        return result
    return wrapper


def save_yaml(savepath, data):
    with open(savepath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, 'rb') as file:
        loaded_file = pickle.load(file)
    return loaded_file

def get_angle_in_0_2pi(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

def make_dir(path, exist_ok=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=exist_ok)

def read_cfg_file(cfg_name, print_dict=False):
    with open(cfg_name, "r") as file:
        try:
            cfg = yaml.full_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            print("*"*20)
    # if print_dict:
    #     print("="*20)
    #     for item in cfg.keys():
    #         print(f"{item}:")
    #         for subitem in cfg[item].keys():
    #             print(f"\t {subitem}: \t {cfg[item][subitem]}")
    #     print("="*20)

    return cfg

class log_and_viz_params:
    def __init__(self, model):
        self.model = model
        self.wb_log_dict = {}
        self.grad_log_dict = {}
        # No. of weights and bias arrays
        self.N = sum(1 for dummy in model.named_parameters()) 
        self.log_counts = 0
        for n_param, param in zip(model.named_parameters(), model.parameters()):
            self.wb_log_dict[n_param[0]] = []
            self.grad_log_dict[n_param[0]] = []
        

    def log_params(self, device=None):
        self.log_counts += 1

        if device=='cuda':
            self.model = self.model.cpu()
            for n_param, param in zip(self.model.named_parameters(), self.model.parameters()):
                self.wb_log_dict[n_param[0]].append(param.cpu().detach().numpy().reshape(-1,1))
                self.grad_log_dict[n_param[0]].append(param.grad.cpu().numpy().reshape(-1,1))

        else:
            for n_param, param in zip(self.model.named_parameters(), self.model.parameters()):
                self.wb_log_dict[n_param[0]].append(param.detach().numpy().reshape(-1,1))
                self.grad_log_dict[n_param[0]].append(param.grad.numpy().reshape(-1,1))



    def print_logs(self, idx=-1):
        return


    def visualize_wb(self, size=4, savefig=None):

        height_ratios = [size for i in range(self.N)] 
        fig, axs = plt.subplots(self.N, 2, gridspec_kw={'height_ratios': height_ratios })
        fig.set_figheight(self.N*size)

        named_pars = self.model.named_parameters()
        pars = self.model.parameters()
        for i, (n_param, param) in enumerate(zip(named_pars, pars)):
            key = n_param[0]
            axs[i,0].set_title(key)
            (odim, idim) = self.wb_log_dict[key][0].shape #should be independent of i
            print("verify odim , idim = ", odim, idim)

            layer_wts= np.array(self.wb_log_dict[key])
            layer_grads = np.array(self.grad_log_dict[key])
            n_wts = odim*idim
            for j in range(n_wts):
                axs[i,0].plot(layer_wts[:,j,0])
                axs[i,1].plot(layer_grads[:,j,0])

        fig.tight_layout()
        if savefig != None:
            plt.savefig(savefig, dpi= 1200)
        
        return
    
def print_dict(dic):
    for key, value in dic.items():
        print(f"{key}:\t {value}")
        

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
def show_num_of_params(model, model_name="transformer", only_trainable=True):
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total {model_name} Params: {num_params}")
    if only_trainable:
        print(f"Trainable {model_name} Params: {trainable_params} ")
        return trainable_params
    return num_params

def convert_dict_to_obj(dictr):
    obj = collections.namedtuple('cfg', dictr.keys())(*dictr.values())
    return obj

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
        save_path = os.path.join(save_dir, "arch.pt")
        torch.save(model, save_path)
        return
    
    # save states
    if save_type == 'cur_best':
        save_path = os.path.join(save_dir, f"best_vloss.pt")
    elif save_type == "some_epoch":
        save_path = os.path.join(save_dir, f"ep{epoch}.pt")
    elif save_type == 'cur_best_translation':
        save_path = os.path.join(save_dir, f"best_translation_loss.pt")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state': scheduler.state_dict(),
            'tr_loss': loss[0],
            'val_loss': loss[1],
            }, save_path)
    
    return

def plot_grad_flow(named_parameters, path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # plt.savefig('/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tmp/grads.png')
    plt.savefig(os.path.join(path, 'gradients.png'))