import torch
from torch import nn

from transformers import ViTMAEConfig, ViTMAEForPreTraining


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing a ViT MAE vit-mae-base style configuration
# timm_VitMae
class fbMae_model(nn.Module):

    def __init__(self, pixel_scale=1):
        super().__init__()
        self.scl = pixel_scale

        # Initializing a model (with pre_trained weights) from the vit-mae-base style configuration
        self.model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')   # pre-trained on imagenet-1K


    def config(self):
        return self.model.config    
    
    
    def forward(self, x, padded=False):
        x = x*self.scl   # pixel scaling

        global device   
        if padded==False:   # padding a ZERO-PIXELLED image channel
            shp = x.shape   
            zero_pad = torch.zeros(shp[0], 1, *shp[2:]).to(device)
            x = torch.cat((x, zero_pad), dim=1)

        outputs = self.model(x)   # feeding input to model

        return outputs
    

# END


