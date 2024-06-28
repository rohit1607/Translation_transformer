config = {

    # preTrained fbMae_model finetuning
    # "tae_path": "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Nov23_100envs_clr_randomr/",
    # "tae_path": "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Jan24_1env_clr_dg3/",
    "tae_path": "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Jan24_debug_dg3multiobs",
    
    # "tae_path": "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tiny_ae_logs/Nov23_100envs_lin_randomr",

    "nt": 10, # time steps (nt must be between 0 and the diffrence of the latter two [fact_1*tts/kf, fact_2*tts/kf]) tts=total_time_steps
    "frac_1":0.25,    # slicing_factor_1 for sine_type_time_variation (CHANGE WITH CARE!!)
    "frac_2":0.75,    # slicing_factor_2 for sine_type_time_variation (CHANGE WITH CARE!!)
    "nr": 1,    # realization steps
    "kf": 2,  # wave number (CHANGE WITH CARE!!)
    "scl": 1, # scale-up factor for pixel values of input datasets
    
    "n_samples": 2,   # for testing.py  +  for aftain_test

    "nth": 1,  # logging image at nth epoch

    "aftrain_testDatpath": "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_scaled_0.7_multi_ran_stat_new_25/",   # must always include
    # "aftrain_testDatpath": "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/DG3/raw_data/nT_120/",
    # "aftrain_testDatpath":'/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/Perlin/Perlin_g50x50x60_1198',
    "Train_data_path": "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/",
    # "Train_data_path": "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/Perlin/",
    # "Train_data_path": "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/DG3/raw_data/",
    "dataset": 'Perlin',
    "image_size": 512,
    "patch_size": 16,    # patch height, patch width
    "obstacle_mask": 'mag',
    # train_param
    "epochs": 100,   

    # loader
    "batch_size": 32,
    "random_seed": 50,
    "train_size": 0.8,
    "val_size": 0.1,
    "test_size": 0.1,
    "plot_dat": False,   # gives info of data used for training or finetuning

    # optimizers & schedulers
    "optimizer": "adam",
    "scheduler": "None",

    # [SGD --optim]
    "lr": 0.0001,
    "momentum": 0.9,
    "wd": 0.001,

    # [ADAM or ADAMW --optim]
    # "lr": config above,
    # "wd": config above,
    
    # [seqlr --optim]
    "final_lr": 0.00001,
    

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




# END
