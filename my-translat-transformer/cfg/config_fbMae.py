config = {

    # preTrained fbMae_model finetuning
    "fbMae_path": "/media/HDD/rohit/Translation_transformer/my-translat-transformer/mae_logs/GPT_MAE/Sept23_100envs_cosw_randomr_hs_1000E/",

    "nt": 10, # time steps (nt must be between 0 and the diffrence of the latter two [fact_1*tts/kf, fact_2*tts/kf]) tts=total_time_steps
    "frac_1":0.25,    # slicing_factor_1 for sine_type_time_variation (CHANGE WITH CARE!!)
    "frac_2":0.75,    # slicing_factor_2 for sine_type_time_variation (CHANGE WITH CARE!!)
    "nr": 1,    # realization steps
    "kf": 2,  # wave number (CHANGE WITH CARE!!)
    "scl": 1, # scale-up factor for pixel values of input datasets
    
    "n_samples": 2,   # for testing.py  +  for aftain_test

    "nth": 3,  # logging image at nth epoch

    "aftrain_testDatpath": "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_scaled_0.7_multi_ran_stat_new_25/",   # must always include

    "Train_data_path": "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/",

    "image_size": 224,
    "patch_size": 16,    # patch height, patch width

    # train_param
    "epochs": 1000,   

    # loader
    "batch_size": 256,
    "random_seed": 50,
    "train_size": 0.8,
    "val_size": 0.1,
    "test_size": 0.1,
    "plot_dat": False,   # gives info of data used for training or finetuning



    # optimizers & schedulers
    "optimizer": "sgd",
    "scheduler": "cosw",

    # [SGD --optim]
    "lr": 0.001,
    "momentum": 0.9,
    "wd": 0.001,

    # [ADAM or ADAMW --optim]
    # "lr": config above,
    # "wd": config above,
    

    # [LIN --sched]
    "opt_factor": 0.99, 
    "patience": 15,

    # [COS --sched]
    # "epochs": config above,
    "eta_min": 0.00001,
    "last_epoch": -1,
    "verbose": False,

    # [COSW --sched]
    "first_cycle_steps": 60,
    "cycle_mult": 1.0,
    "min_lr": 1.5e-6,
    "max_lr": 0.001,
    "warmup_steps": 8,
    "gamma": 1.0,



    # continue training params
    # use following cmd---> 
    # [python continue_training.py --mode finetuned]
    "continue_model": "Bestuned_fbMae_Model_17D_5E",

    "load_best_model": "Bestuned_fbMae_Model_17D_5E",
    
    "checkpoints_name": "checkpoints_fbMae_model",

    "cont_path": "/home/rohit/Documents/Research/data_prep/raj/fbMae_logs/",

}




hpt_config = {

    "method": "random",

    "name": "HPTuning_FBMAE",

    "metric": {"name":"Best Val Loss", "goal":"minimize"},

    "parameters": {

        # preTrained fbMae_model finetuning
        "fbMae_path": {"value": "/home/rohit/Documents/Research/data_prep/raj/fbMae_logs/"},

        "nt": {"value": 10}, # time steps (nt must be between 0 and the diffrence of the latter two [fact_1*tts/kf, fact_2*tts/kf]) tts=total_time_steps
        "frac_1": {"value": 0.25},    # slicing_factor_1 for sine_type_time_variation (CHANGE WITH CARE!!)
        "frac_2": {"value": 0.75},    # slicing_factor_2 for sine_type_time_variation (CHANGE WITH CARE!!)
        "nr": {"value": 1},    # realization steps
        "kf": {"value": 2},  # wave number (CHANGE WITH CARE!!)
        "scl": {"value": 10}, # scale-up factor for pixel values of input datasets
        
        "n_samples": {"value": 5},   # for testing.py  +  for aftain_test

        "nth": {"value": 3},  # logging image at nth epoch

        "aftrain_testDatpath": {"value": "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/GenHW_TV_DNV_dl_c83_m10_s20_f2_A1/"},   # must always include
        
        "test_path": {"value": "/home/rohit/Documents/Research/data_prep/raj/test_logs/"},

        "Train_data_path": {"value": "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/"},

        # train_param
        "epochs": {"value": 2},   

        # loader
        "batch_size": {"value": 256},
        "random_seed": {"value": 50},
        "train_size": {"value": 0.8},
        "val_size": {"value": 0.1},
        "test_size": {"value": 0.1},
        "plot_dat": {"value": False},   # gives info of data used for training or finetuning

        "image_size": {"value": 224},   
        "patch_size": {"value": 16},    

        # Optimizers & Schedulers
        "optimizer": {"value": "sgd"},
        "scheduler": {"values": ["lin", "cos"]},


        # [SGD --optim]
        "lr": {"value": 0.001},  
        "wd": {"value": 0.005},
        "momentum": {"value": 0.9},

        # [ADAM or ADAMW --optim]
        # "lr": config above,
        # "wd": config above,


        # [LIN --sched]
        "opt_factor": {"value": 0.99}, 
        "patience": {"value": 15},

        # [COS --sched]
        # "epochs": config above,
        "eta_min": {"value": 0},
        "last_epoch": {"value": -1},
        "verbose": {"value": False},
        "T_max" : {"value":500},

        # [COSW --sched]
        "first_cycle_steps": {"value": 60},
        "cycle_mult": {"value": 1.0},
        "min_lr": {"value": 1.5e-6},
        "max_lr": {"value": 0.001},
        "warmup_steps": {"value": 8},
        "gamma": {"value": 1.0}
    }
}

# END
