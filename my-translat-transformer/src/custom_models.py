"""
Modified or original code

"""

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from transformers import ViTMAEConfig, ViTMAEForPreTraining
from transformers import ViTConfig, ViTModel

import timm

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        # print(f"in posEnc.init emb_size = {emb_size}")

        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor, timesteps: Tensor):
        # print(f"in posEnc.forward token_embedding.shape = {token_embedding.shape},\n self.pos_embedding.shape = {self.pos_embedding.shape}, {token_embedding.size(0)}")
        emb = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :])
        return emb
                                # [64, 70, 8]            (70,8)

class SimplePositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len):
        super(SimplePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, emb_size)
        
        # TO DO : Raj to check correctness
        # emb_vec = emb_vec.detach().cpu().numpy()
        # emb_vec = torch.from_numpy(emb_vec.reshape((-3,)+emb_vec.shape[:2]).transpose(1,2,0))
        
    def forward(self, emb_vec, timesteps):
        emb = self.pos_embedding(timesteps) + emb_vec
        # emb = self.pos_embedding(timesteps) + torch.mean(emb_vec, dim=2)

        return emb

# # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size

#     def forward(self, tokens: Tensor):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class LinTokenEmbedding(nn.Module):
    def __init__(self, ip_vec_dim: int, emb_size):
        super(LinTokenEmbedding, self).__init__()
        self.embedding = nn.Linear(ip_vec_dim, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        a= self.embedding(tokens.to(torch.float32)) * math.sqrt(self.emb_size)  
        return a

# Seq2Seq Network
class mySeq2SeqTransformer_v1(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vec_dim: int,
                 tgt_vec_dim: int,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 max_len: int = 120,
                 batch_first = True,
                 positional_encoding: str = "simple",
                 ):
        super(mySeq2SeqTransformer_v1, self).__init__()

        if dim_feedforward == None:
            dim_feedforward = nhead * emb_size
 
        self.max_len = max_len
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=batch_first)
        self.generator = nn.Sequential(nn.Linear(emb_size, tgt_vec_dim), nn.Sigmoid())
        self.src_tok_emb = LinTokenEmbedding(src_vec_dim, emb_size)
        self.tgt_tok_emb = LinTokenEmbedding(tgt_vec_dim, emb_size)
        if positional_encoding == 'sin':
            self.positional_encoding = PositionalEncoding(
                emb_size, dropout=dropout, maxlen=max_len)
        elif positional_encoding == "simple":
            self.positional_encoding = SimplePositionalEncoding(emb_size, max_len)
        else:
            raise ValueError("No such positional_encoding type")

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                timesteps: Tensor):
        # print(f"*** in myseq.forward src.shape = {src.shape},{src.dtype}, {trg.shape}, {trg.dtype} ")
        #  # positional embedding usage as in pytorch translation tutorial
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        
        #  simple positional embedding as in Decistion Transformer
        src_emb = self.positional_encoding(self.src_tok_emb(src), timesteps)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg), timesteps)
        # print(f"src and tgt shapes ; {src_emb.shape}, {tgt_emb.shape}")
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, timesteps: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src), timesteps), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, timesteps: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt), timesteps), memory,
                          tgt_mask)




class e2e_Seq2SeqTransformer_v1(nn.Module):
    """
    model for end-to-end training. Comprises:
    - ViTMAE for encoding representations
    - 
    """
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vec_dim: int,
                 tgt_vec_dim: int,
                 device: str,
                 preprocessor: str, # 'ViTMAE' or 'ViT' or 'ViT_scratch' or 'mobilenetv3
                 preprocessor_cfg: dict,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 max_len: int = 120,
                 batch_first = True,
                 positional_encoding: str = "simple",
                 pixel_scale = 10,
                
                 
                 ):

        super(e2e_Seq2SeqTransformer_v1, self).__init__()

        if dim_feedforward == None:
            dim_feedforward = nhead * emb_size
        self.device = device
        self.max_len = max_len
        self.preprocessor = preprocessor
        self.pp_cfg = preprocessor_cfg
        if preprocessor == 'ViTMAE':
            self.mae = fbMae_model(pixel_scale=pixel_scale)
        elif preprocessor == 'ViT':
            self.vit = timm_ViT(pixel_scale=pixel_scale)
            # get self.vit specific transforms (normalization, resize)
            data_config = timm.data.resolve_model_data_config(self.vit)
            self.vit_transforms = timm.data.create_transform(**data_config, is_training=True)
        elif preprocessor == 'ViT_scratch':

            # Initializing a ViT vit-base-patch16-224 style configuration
            configuration = ViTConfig( hidden_size = self.pp_cfg['hidden_size'],
                                      num_hidden_layers = self.pp_cfg['num_hidden_layers'],
                                      num_attention_heads = self.pp_cfg['num_attention_heads'],
                                      intermediate_size = self.pp_cfg['intermediate_size'],
                                      hidden_dropout_prob = self.pp_cfg['hidden_dropout_prob'],
                                      attention_probs_dropout_prob = self.pp_cfg['attention_probs_dropout_prob'],
                                      image_size = self.pp_cfg['image_size'],
                                      patch_size = self.pp_cfg['patch_size'],
                                      num_channels = self.pp_cfg['num_channels'],
                                      )

            # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
            self.vit = ViTModel(configuration)

            # Accessing the model configuration
            self.vit_cfg = self.vit.config
        
        # elif preprocessor == 'mobilenetv3':
        #     self.mobilenetv3 = mobilenetv3(pixel_scale=pixel_scale)
        #     data_config = timm.data.resolve_model_data_config(self.mobilenetv3)
        #     self.mobilenetv3_transforms = timm.data.create_transform(**data_config, is_training=False)
            
            
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=batch_first)
        self.generator = nn.Sequential(nn.Linear(emb_size, tgt_vec_dim), nn.Sigmoid())
        self.src_tok_emb = LinTokenEmbedding(src_vec_dim, emb_size)
        self.tgt_tok_emb = LinTokenEmbedding(tgt_vec_dim, emb_size)
        if positional_encoding == 'sin':
            self.positional_encoding = PositionalEncoding(
                emb_size, dropout=dropout, maxlen=max_len)
        elif positional_encoding == "simple":
            self.positional_encoding = SimplePositionalEncoding(emb_size, max_len)
        else:
            raise ValueError("No such positional_encoding type")
    
    def process_imgs_with_mae(self, img_seq: Tensor, loc: Tensor):
        """
        - passes img_seq as a batch through mae encoder,
        - adds location embedding to temporal sequence
        returns src- the input the translation model
        """
        # reshape src (seq of images) B,T,... -> BxT, ...
        B,T,C,H,W = img_seq.shape
        # img_seq = img_seq.reshape((B*T,C,H,W)) 
        # img_seq = img_seq[0:10]
        mae_enc_op = []
        print(f"img_seq.shape = {img_seq.shape}")
        for img_seq_sample in img_seq:
            mae_outputs = self.mae(img_seq_sample) # expected shape -> (B*T,dim)
            # mae_outputs.hidden_states: -1 for last layer, [B,50(masking),dim] (B,50,768)
            mae_enc_op.append(mae_outputs.hidden_states[-1][:,0,:]) # 0 for clf token
        mae_enc_op = torch.stack(mae_enc_op, axis=0) # shape: B,T,dim
        # mae_enc_op = mae_enc_op.reshape((B,T,-1))
        # TODO: can do padding with outside the function
        loc_pad = torch.zeros((B, mae_enc_op.shape[-1] - loc.shape[-1])).to(self.device)
        loc_tok = torch.cat([loc, loc_pad], axis=-1)
        loc_tok = torch.unsqueeze(loc_tok, 1)
        src = torch.cat([loc_tok, mae_enc_op], axis=1)
        
        return src
    
    def process_imgs_with_vit(self, img_seq: Tensor, loc: Tensor):
        # torch.Size([1, 3, 224, 224])
        B,T,C,H,W = img_seq.shape
        img_seq = img_seq.reshape((B*T,C,H,W)) 
        # self.vit.eval() #TODO: when do we use eval() and train)()
        output = self.vit(img_seq)  # output is (batch_size, num_features) shaped tensor
        output = output.reshape((B,T,-1))
        loc_pad = torch.zeros((B, output.shape[-1] - loc.shape[-1])).to(self.device)
        loc_tok = torch.cat([loc, loc_pad], axis=-1)
        loc_tok = torch.unsqueeze(loc_tok, 1)
        src = torch.cat([loc_tok, output], axis=1)
        
        return src
    
    def process_imgs_with_vit_scratch(self, img_seq: Tensor, loc: Tensor):

        B,T,C,H,W = img_seq.shape
        img_seq = img_seq.reshape((B*T,C,H,W)) 
        # self.vit.eval()
        output = self.vit(img_seq)  # output is (batch_size, num_features) shaped tensor
        output = output.last_hidden_state[:,0]
        output = output.reshape((B,T,-1))
        loc_pad = torch.zeros((B, output.shape[-1] - loc.shape[-1])).to(self.device)
        loc_tok = torch.cat([loc, loc_pad], axis=-1)
        loc_tok = torch.unsqueeze(loc_tok, 1)
        src = torch.cat([loc_tok, output], axis=1)
        
        return src
    
    # def process_imgs_with_mobilenetv3(self, img_seq: Tensor, loc: Tensor):
        


    def forward(self,
                img_seq: Tensor, #shape: B,T,C,H,W
                loc: Tensor, #shape B,4
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                timesteps: Tensor):
        
        if self.preprocessor == 'ViTMAE':
            src = self.process_imgs_with_mae(img_seq, loc)
        elif self.preprocessor == 'ViT':
            src = self.process_imgs_with_vit(img_seq, loc)
        elif self.preprocessor == 'ViT_scratch':
            src = self.process_imgs_with_vit_scratch(img_seq, loc)
        # elif self.preprocessor == 'mobilenetv3':
        #     src = self.process_imgs_with_mobilenetv3(img_seq, loc)
        #  simple positional embedding as in Decistion Transformer
        src_emb = self.positional_encoding(self.src_tok_emb(src), timesteps)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg), timesteps)
        
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, img_seq: Tensor, loc: Tensor, src_mask: Tensor, timesteps: Tensor):
        if self.preprocessor == 'ViTMAE':
            src = self.process_imgs_with_mae(img_seq, loc)
        elif self.preprocessor == 'ViT':
            src = self.process_imgs_with_vit(img_seq, loc)   
        elif self.preprocessor == 'ViT_scratch':
            src = self.process_imgs_with_vit_scratch(img_seq, loc)
        # elif self.preprocessor == 'mobilenetv3':
        #     src = self.process_imgs_with_mobilenetv3(img_seq, loc)
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src), timesteps), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, timesteps: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt), timesteps), memory,
                          tgt_mask)




class timm_ViT(nn.Module):
    
    def __init__(self, pixel_scale=1):
        super().__init__()
        self.scl = pixel_scale
        self.model = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0)
        
    def config(self):
        return self.model.config
    
    def forward(self, x, pad_channel3=True):
        """
        pad_channel3 needs to be True for vx_vy images which only has 2 channels. 
        However, pretrained MAE needs 3 channel images. Hence padding is required.
        
        When training with obstacles masks, it will not be required,
        i.e. pad_channel3 can be set to False
        """
        x = x*self.scl   # pixel scaling

        if pad_channel3:   # padding a ZERO-PIXELLED image channel
            # shp = x.shape   
            # zero_pad = torch.zeros(shp[0], 1, *shp[2:]).to(device)
            pad = x[:,0].unsqueeze(1)
            pad = pad*0
            x = torch.cat((x, pad), dim=1)

        outputs = self.model(x)   # feeding input to model

        return outputs        


class mobilenetv3(nn.Module):
    
    def __init__(self, pixel_scale=1):
        super().__init__()
        self.scl = pixel_scale
        self.model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True, num_classes=0)
        
    def config(self):
        return self.model.config
    
    def forward(self, x, pad_channel3=True):
        """
        pad_channel3 needs to be True for vx_vy images which only has 2 channels. 
        However, pretrained MAE needs 3 channel images. Hence padding is required.
        
        When training with obstacles masks, it will not be required,
        i.e. pad_channel3 can be set to False
        """
        x = x*self.scl   # pixel scaling

        if pad_channel3:   # padding a ZERO-PIXELLED image channel
            # shp = x.shape   
            # zero_pad = torch.zeros(shp[0], 1, *shp[2:]).to(device)
            pad = x[:,0].unsqueeze(1)
            pad = pad*0
            x = torch.cat((x, pad), dim=1)

        outputs = self.model(x)   # feeding input to model

        return outputs   

# --------------------------------------------------------------------------------------------------------------------------------
#                       MAE models
# --------------------------------------------------------------------------------------------------------------------------------


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing a ViT MAE vit-mae-base style configuration
# timm_VitMae
class fbMae_model(nn.Module):
    
    def __init__(self, pixel_scale=1):
        super().__init__()
        self.scl = pixel_scale

        # Initializing a model (with pre_trained weights) from the vit-mae-base style configuration
        # Initializing a model (with pre_trained weights) from the vit-mae-base style configuration
        self.model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')   # pre-trained on imagenet-1K
        PreTrained_wandb = self.model.state_dict()
        self.model = ViTMAEForPreTraining(ViTMAEConfig(output_hidden_states=True))
        self.model.load_state_dict(PreTrained_wandb)

    def config(self):
        return self.model.config    
    
    
    def forward(self, x, pad_channel3=True):
        """
        pad_channel3 needs to be True for vx_vy images which only has 2 channels. 
        However, pretrained MAE needs 3 channel images. Hence padding is required.
        
        When training with obstacles masks, it will not be required,
        i.e. pad_channel3 can be set to False
        """
        x = x*self.scl   # pixel scaling

        if pad_channel3:   # padding a ZERO-PIXELLED image channel
            # shp = x.shape   
            # zero_pad = torch.zeros(shp[0], 1, *shp[2:]).to(device)
            pad = x[:,0].unsqueeze(1)
            pad = pad*0
            x = torch.cat((x, pad), dim=1)

        outputs = self.model(x)   # feeding input to model

        return outputs
    

# END


