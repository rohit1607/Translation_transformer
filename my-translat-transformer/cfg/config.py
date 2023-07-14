config = {

    "nt": 30, # time steps (nt must be between 0 and the diffrence of the latter two [fact_1*tts/kf, fact_2*tts/kf]) tts=total_time_steps
    "frac_1":0.25,    # slicing_factor_1 for sine_type_time_variation (CHANGE WITH CARE!!)
    "frac_2":0.75,    # slicing_factor_2 for sine_type_time_variation (CHANGE WITH CARE!!)
    "nr": 100,    # realization steps
    "kf": 2,  # wave number (CHANGE WITH CARE!!)
    "scl": 10, # scale-up factor for pixel values of input datasets
    
    "n_samples": 10,   # for testing.py  +  for aftain_test

    "nth": 150,  # for train_mae.py 

    "aftrain_testDatpath": "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/GenHW_TV_DNV_dl_c83_m10_s20_f2_A1/",   # must always include
    
    "test_path": "/home/rohit/Documents/Research/data_prep/raj/test_logs/",
    "train_path": "/home/rohit/Documents/Research/data_prep/raj/train_logs/",

    "Train_data_path": "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/",


    # VIT
    "image_size": 224,
    "patch_size": 16,    # patch height, patch width
    "num_classes": 0, # useless
    "dim": 512,
    "depth": 8,
    "heads": 8,
    "mlp_dim": 512, #  ffn dim that lies at the end of every encoder unit
    "dim_head": 64,     #  K, finally which is used in attention formula


    # MAE
    "masking_ratio": 0.75,   # the paper recommended 75% masked patches
    "decoder_dim": 512,      # paper showed good results with just 512
    "decoder_depth": 8,       # anywhere from 1 to 8 
    "decoder_heads": 8,
    "decoder_dim_head": 64,    # K, finally which is used in attention formula


    # train_param
    "lr": 0.001,  # see literature for appropriate lr and lr scheduler
    "wd": 0.00,    # see literatrue for apt value 
    "epochs": 500,   
    "opt_factor": 0.99, # rohit: see in code
    "patience": 20,


    # loader
    "batch_size": 256,
    "val_size": 0.1,
    "loader_path": "/home/rohit/Documents/Research/data_prep/raj/loader_logs/",





    # timm logs path
    "timm_path": "/home/rohit/Documents/Research/data_prep/raj/timm_mae_logs/",


    # timm_VitMae
    "hidden_size": 512,    # encoder_dim
    "num_hidden_layers": 8,    # encoder_depth
    "num_attention_heads": 8,    # encoder_heads
    "intermediate_size": 512,    # encoder_ffn out_dim
    "hidden_act": 'gelu',    # ffn activation_fxn
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    # "image_size": 224,    # configure from above key
    # "patch_size": 16,    # configure from above key
    "num_channels": 2,
    "qkv_bias": False,
    "decoder_num_attention_heads": 8,    # decoder_heads
    "decoder_hidden_size": 512,    # decoder_dim
    "decoder_num_hidden_layers": 8,    # decoder_depth
    "decoder_intermediate_size": 512,
    "mask_ratio": 0.75,
    "norm_pix_loss": False
}


# Hyperparameters Tuning...

hpt_config = {

    "method": "random",

    "name": "HPTuning_MAE",

    "metric": {"name":"Best Val Loss", "goal":"minimize"},

    "parameters": {

        "nth": {"value": 150},  # for train_mae.py
        
        "test_path": {"value": "/home/rohit/Documents/Research/data_prep/raj/test_logs/"},
        "train_path": {"value": "/home/rohit/Documents/Research/data_prep/raj/train_logs/"},
  

        # VIT
        "image_size": {"value": 224},
        "patch_size": {"value": 16},    # patch height, patch width
        "num_classes": {"value": 0},    # useless
        "dim": {"value": 32},
        "depth": {"values": [8, 2]},
        "heads": {"values": [8, 2]},
        "mlp_dim": {"value": 32},
        "dim_head": {"value": 64},

        #MAE
        "masking_ratio": {"value": 0.75},   # the paper recommended 75% masked patches
        "decoder_dim": {"value": 64},      # paper showed good results with just 512
        "decoder_depth": {"value": 1},       # anywhere from 1 to 8
        "decoder_heads": {"values": [2, 8]},
        "decoder_dim_head": {"value": 64},



        # train_param
        "lr": {"min": 0.00000001, "max": 0.001},
        "wd": {"value": 0.01},
        "epochs": {"value": 100},
        "opt_factor": {"value":0.95},
        "patience": {"value": 7},


        # loader 
        "batch_size": {"values": [256, 128, 64]},
        "val_size": {"value": 0.1},
        "loader_path": {"value": "/home/rohit/Documents/Research/data_prep/raj/loader_logs/"}
        
    }
}


# END

