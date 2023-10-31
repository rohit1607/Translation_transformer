import torch
from torch import nn

from transformers import ViTMAEConfig, ViTMAEForPreTraining


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing a ViT MAE vit-mae-base style configuration
# timm_VitMae
class fbMae_model(nn.Module):
    def __init__(self, pixel_scale=1, output_hidden_states=True):
        super().__init__()
        self.scl = pixel_scale

        # Initializing a model (with pre_trained weights) from the vit-mae-base style configuration
        self.model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base',
                                                          output_hidden_states=output_hidden_states)   # pre-trained on imagenet-1K


    def config(self):
        return self.model.config    
    
    
    def forward(self, x):
        x = x*self.scl   # pixel scaling
        
        outputs = self.model(x)   # feeding input to model
        return outputs
    

