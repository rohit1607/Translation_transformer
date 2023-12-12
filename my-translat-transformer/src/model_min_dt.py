"""
This implemntation is derived from min-decision-transformer

"""

import math
# from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import seaborn as sns
import sys
from src_utils import generate_square_subsequent_mask

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        """
        h_dim: hidden dimensions
        max_T: TODO
        n_heads: no. of multi-attenion heads
        drop_p: dropout probability
        """

        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T
        C = h_dim*n_heads

        self.q_net = nn.Linear(C, C)
        self.k_net = nn.Linear(C, C)
        self.v_net = nn.Linear(C, C)
        self.proj_net = nn.Linear(C, C)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)


    def forward(self, x):
        B, T, C = x.shape   # batchsize, max_t, h_dim for n_head=1


        N, D = self.n_heads, torch.div(C, self.n_heads, rounding_mode='floor')# N = num heads, D = attention dim
         

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)
        # print(f'in attention q shape: {q.shape}')

        # weights (B, N, T, T)  <-- B,N,T,D  @  B,N,D,T
        weights = torch.matmul(q, k.transpose(2,3)) / math.sqrt(float(k.size(-1)))

        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        self.attention_weights = weights
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)
        # plot_attention_weights(weights)
        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))


        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        C = h_dim*n_heads
        self.mlp = nn.Sequential(
                nn.Linear(C, 4*C),
                nn.GELU(),
                nn.Linear(4*C, C),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, target_token=False):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.n_heads = n_heads
        ### transformer blocks
        if target_token:
            input_seq_len = (3 * context_len) +1 
        else:
            input_seq_len = 3 * context_len

        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.blocks = blocks
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.merge_heads_linear = torch.nn.Linear(h_dim*n_heads, h_dim)
        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = False # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go, target_token=None):
        # TODO: shubham: add env_seq at embedding
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        # TODO: Shubham
        # env_embeddings =  self.mae() + time_emb
        # Note: way 1: make self.Embed_env and do  self.embed_env(env_op_from_mae) + time_emb
        #        way 2: make (forve) the emb_dim = the emb_dim_from_mae    < although dt emb_dim can become too large
          
        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        
        # TODO: Shubham 
        # make changes to incorporate env sequences
        # (r_0, s_0, e0, a_0, r_1, e1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)


        # print(f"h.shape = {h.shape} \n states.shape = {states.shape}")
        if target_token!=None:
            # print(f"target_token.shape =  {target_token.shape}")
            target_token = torch.unsqueeze(target_token, 1)
            target_st_embeddings = self.embed_state(target_token)
            # print(f" target_st_embeddings.shape = {target_st_embeddings.shape}")
            h = torch.cat((target_st_embeddings,h), axis=1)
            # print(f"h.shape = {h.shape} ")

        #     # target token == target position 
        #     target_state = [35, target_token[0], target_token[1]]
        #     target_states = []
        #     target_state_embedding = self.embed_state()
        #     pass
        # sys.exit()

        h = self.embed_ln(h)

        # myedit for making multihead work  
        h = h.repeat((1,1,self.n_heads))


        # transformer and prediction
        # print(f"pre-transf h.shape = {h.shape} ")
        h = self.transformer(h)
        # print(f"post-transf h.shape = {h.shape} ")
        h = self.merge_heads_linear(h)
        # print(f"post-merge h.shape = {h.shape} ")
        # sys.exit()
        if target_token!=None:
            h = h[:,1:,:]   # B x (3T + 1) x hdim  --> B x (3T ) x hdim  exclude target tokem
            # print(f"post prune target token h.shape = {h.shape} ")

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        # print(f"post reshape h.shape = {h.shape} ")

        # get predictions
        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s
        # print(f"h[:,2].shape =  {h[:,2].shape}")
        # print(f"state_preds in model =  {state_preds.shape}")

        return state_preds, action_preds, return_preds
    

class Transformer_decoder(nn.Module):
    
    def __init__(self, state_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, target_token=False):
        """
        state referes to env embedding from mae encoder
        """
        super().__init__()

        self.state_dim = state_dim
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        ### transformer blocks
        blocks = [Block(h_dim, context_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.blocks = blocks
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        #TODO: Rohit: add cosine option
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.merge_heads_linear = torch.nn.Linear(h_dim*n_heads, h_dim)
        ### prediction heads
        self.predict_state = torch.nn.Linear(h_dim, state_dim)


    def forward(self, timesteps, states):
        # TODO: shubham: add env_seq at embedding
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        h = self.embed_state(states) + time_embeddings

        # ( e0, e1, s_1, a_1,  ... e_{N-1} , e_N)
        #  |---- input -------------------|
        #       |---- output -------------------| 
        h = self.embed_ln(h)

        # myedit for making multihead work  
        h = h.repeat((1,1,self.n_heads))
        
        # transformer and prediction
        h = self.transformer(h)
        h = self.merge_heads_linear(h)
        env_preds = self.predict_state(h)  
       
        return env_preds
    
    
class Transformer_causal_encoder(nn.Module):
    def __init__(self, inp_dim, n_blocks, emb_dim, context_len, n_heads, drop_p=0.1, ff_dim=None, device='cpu'):
        super().__init__()
        self.inp_dim = inp_dim
        self.n_blocks = n_blocks
        self.emb_dim = emb_dim
        self.context_len = context_len
        self.n_heads = n_heads
        self.drob_p = drop_p
        self.ff_dim = ff_dim if ff_dim is not None else context_len*4
        self.embed_timestep = nn.Embedding(context_len, emb_dim)
        self.embed_inp = nn.Linear(inp_dim, emb_dim)
        self.enc_layer = torch.nn.TransformerEncoderLayer(d_model=emb_dim,
                                                          nhead=n_heads,
                                                          dim_feedforward=self.ff_dim,
                                                          dropout=drop_p,
                                                          activation='gelu',
                                                          batch_first=True,
                                                          )
        self.model = torch.nn.TransformerEncoder(self.enc_layer, n_blocks)
        self.predictor = nn.Linear(emb_dim, inp_dim)
        self.causal_mask = generate_square_subsequent_mask(context_len, device)
        
    def forward(self, timesteps, src, padding_mask):
        src_emb =  self.embed_inp(src) + self.embed_timestep(timesteps)
        return self.predictor(self.model(src_emb, src_key_padding_mask=padding_mask, mask=self.causal_mask))
    
    
    
# from src_utils import generate_square_subsequent_mask
# inp_dim, n_blocks, emb_dim, context_len, n_heads = 100, 1, 20, 100, 4
# causal_mask = generate_square_subsequent_mask(context_len,'cpu')
# tce =  Transformer_causal_encoder(inp_dim, n_blocks, emb_dim, context_len, n_heads)
# src = torch.cat([torch.zeros(1,context_len//2, inp_dim), torch.ones(1,context_len//2, inp_dim)],axis=1)
# timesteps = torch.range(0,context_len-1,1).type(torch.int32)
# padding_mask = torch.zeros((1,context_len)).type(torch.bool)
# outputs =  tce(src, timesteps, padding_mask, causal_mask)
# print()