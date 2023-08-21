import torch
from torch import nn

from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining


# Initializing a ViT MAE vit-mae-base style configuration
# timm_VitMae
class VitMae_model(nn.Module):

    def __init__(self, model_config, pixel_scale=1):
        super().__init__()
        self.scl = pixel_scale
        self.cfg = model_config
        self.configuration = ViTMAEConfig(
                                hidden_size = self.cfg["hidden_size"],    # encoder_dim
                                num_hidden_layers = self.cfg["num_hidden_layers"],    # encoder_depth
                                num_attention_heads = self.cfg["num_attention_heads"],    # encoder_heads
                                intermediate_size = self.cfg["intermediate_size"],    # encoder_ffn out_dim
                                hidden_act = self.cfg["hidden_act"],    # ffn activation_fxn
                                hidden_dropout_prob = self.cfg["hidden_dropout_prob"],
                                attention_probs_dropout_prob = self.cfg["attention_probs_dropout_prob"],
                                initializer_range = self.cfg["initializer_range"],
                                layer_norm_eps = self.cfg["layer_norm_eps"],
                                image_size = self.cfg["image_size"],
                                patch_size = self.cfg["patch_size"],
                                num_channels = self.cfg["num_channels"],
                                qkv_bias = self.cfg["qkv_bias"],
                                decoder_num_attention_heads = self.cfg["decoder_num_attention_heads"],    # decoder_heads
                                decoder_hidden_size = self.cfg["decoder_hidden_size"],    # decoder_dim
                                decoder_num_hidden_layers = self.cfg["decoder_num_hidden_layers"],    # decoder_depth
                                decoder_intermediate_size = self.cfg["decoder_intermediate_size"],
                                mask_ratio = self.cfg["mask_ratio"],
                                norm_pix_loss = self.cfg["norm_pix_loss"],
                                output_hidden_states = True
                                )
        

        # Initializing a model (with random weights) from the vit-mae-base style configuration
        self.model = ViTMAEForPreTraining(self.configuration)

    def config(self):
        return self.model.config    
    
    def forward(self, x):
        self.x = x*self.scl
        outputs = self.model(self.x)
        return outputs
    

# END


