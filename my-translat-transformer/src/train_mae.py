import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Returns image size in the form (t, t)
def pair(t):                                        
    return t if isinstance(t, tuple) else (t, t)


# Calculates layer normalization before getting into 'Attention' or 'Feed-forward nn'
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

# Feed-forward NN as in the encoder 
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

# Multi-head self-attention as in the encoder
# b=batch_size  h=head_number  n=patch_number  d=feature_number
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # As we only need to learn the weight matrix (same being proposed in the paper)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  
        return self.to_out(out)    


# Transformer (encoders) implemented
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x  


# ViT(input=cls_token+x,  Transformer(encoders),  mlp_head(linear layer to throw number of classes as output))
# b=batch_size  c=channel_number  (h p1)=total image height  (w p2)=total image width  (h w)=number_of_patches
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 2, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(        # converts batched_images_data to batched_patched-images_data dimension
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # h,w = 8; p1,p2 = 32; c = 3
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)      # from here this goes to encoder

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # Query shape
        x = self.to_latent(x)
        return self.mlp_head(x)  


# Masked auto encoder implementation
class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,    # number of transformer
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()     # IF DIMENSIONS ARE NoT EQUAL
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        patches = self.to_patch(img)           # we have image patches (b c H W -> b n p)  p=patch_dim=c*p1*p2
        batch, num_patches, *_ = patches.shape

        tokens = self.patch_to_emb(patches)      # 768 dimension token now you have (b n p -> b n d)  d=dim
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]   # position emb addition
        
        self.tokens_blatent = tokens

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)    #batches also got shuffled here 
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]  # MASKING BEING DONE HERE
        
        self.unmasked = unmasked_indices

        batch_range = torch.arange(batch, device = device)[:, None]   # that's we created this
        tokens = tokens[batch_range, unmasked_indices]   # unmasked token batchwise

        masked_patches = patches[batch_range, masked_indices]     # this are also token only but we are seperating out them from unmasked

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)  # applied only on unmasked (tokens)  [encoder operation]

        decoder_tokens = self.enc_to_dec(encoded_tokens)   # dimension matching encoder ouptut to decoder input      

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)   # adding position embedding to input to decoder

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)  # mask_token are randn that will be fed to decoder
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens         
        decoded_tokens = self.decoder(decoder_tokens)       # applied both on encoded token (unmasked) and random toekn (masked) [decoder operation]

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]     # decoded_tokens is list of all the patches from decoder 
        pred_pixel_values = self.to_pixels(mask_tokens)
        
        self.img_patches = self.to_pixels(decoded_tokens)

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss
    
    def final(self):
        return self.img_patches
    
    def repre_latent(self, img):
        device = img.device

        patches = self.to_patch(img)           # we have image patches (b c H W -> b n p)  p=patch_dim=c*p1*p2
        batch, num_patches, *_ = patches.shape

        tokens = self.patch_to_emb(patches)      # 768 dimension token now you have (b n p -> b n d)  d=dim
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]   # position emb addition
        
        self.tokens_blatent = tokens
        return self.encoder.transformer(self.tokens_blatent)
    
    def unmasked_ind(self):
        return self.unmasked        
    

def extract_velocity(vel_field_data, t, rzn, as_image=True):
    # all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field_data
    #    0          1           2           3           4
    # vx = all_u_mat[t,i,j] + np.matmul(all_ui_mat[t, :, i,j], all_Yi[t, rzn,:])
    # vy = all_v_mat[t,i,j] + np.matmul(all_vi_mat[t, :, i,j], all_Yi[t, rzn,:])
    nmodes =  vel_field_data[2].shape[1]
    vx = vel_field_data[0][t,:,:]
    vy = vel_field_data[1][t,:,:] 

    for m in range(nmodes):
        vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

    if as_image:
        im = np.stack([vx,vy], axis=-1)
        return im
    else:
        return vx,vy
    

def load_vel(data_path):
    all_u_mat = np.load(data_path +'all_u_mat.npy')
    all_ui_mat = np.load(data_path +'all_ui_mat.npy')
    all_v_mat = np.load(data_path +'all_v_mat.npy' )
    all_vi_mat = np.load(data_path +'all_vi_mat.npy')
    all_Yi = np.load(data_path +'all_Yi.npy' )
    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
    return vel_field_data


def plot_vel_field(vx_grid, vy_grid, g_strmplot_lw=1, g_strmplot_arrowsize=1):
    # Make modes the last axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    xlim, ylim = vx_grid.shape
    Xs = np.arange(0,xlim) + 0.5
    Ys = np.arange(0,ylim) + 0.5
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    im = plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)
    plt.savefig('flow')
    return im
    
class VelocityDataset(Dataset):
    def __init__(self, vel_field_data):
        self.vel_field_data = vel_field_data
    
    def __len__(self):
        return 120 * 5      # change the number accordingly
    
    def __getitem__(self, idx):
        rzn = idx // 120
        t = idx % 120
        im = extract_velocity(self.vel_field_data, t, rzn)
        
        im_tensor = torch.tensor(im)
        im_tensor = im_tensor.permute(2,0,1)    #2,100,100
        im_tensor = im_tensor.unsqueeze(0)  # Add batch dimension
        im_tensor = F.interpolate(im_tensor, size=(256, 256), mode='bilinear', align_corners=False)     #1,2,256,256
        im_tensor = im_tensor.squeeze(0)  # Remove batch dimension              
        
        return im_tensor
    


if __name__ == "__main__":

    data_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/data/GenHW/GenHW_TV_DNV_dl_c-8_m5_s20_f2_A1/"
    #OUTDAT= "/home/rohit/Documents/data_prep/raj/genHW_dat.txt"

    vel_field_data = load_vel(data_path)
    print(vel_field_data[4].shape)

    # FILL IN t and r for vel_feild at time t and rzn r
    vel_t_r = extract_velocity(vel_field_data, t=80, rzn=10)
    print(vel_t_r.shape)

    batch_size = 16
    dataset = VelocityDataset(vel_field_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(dataloader)
    print(dataset[3].shape)

    v = ViT(
    image_size = 256,
    patch_size = 32,    # patch height, patch width
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
    )

    mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
    )

    # without scheduler
    optimizer = torch.optim.Adam(mae.parameters(), lr=0.00005)

    mae.to(device)
    if device.type == "cuda":
        print("Using CUDA")
    else:
        print("Not using CUDA")
        
    loss_epochwise = []

    for epoch in range(10):
        for batch_idx, batch in enumerate(dataloader):
            images = batch.to(device)

            optimizer.zero_grad()

            outputs = mae(images)

            loss = outputs.mean()

            loss.backward()

            optimizer.step()
        
        loss_epochwise.append(loss.item())
        print("epoch",epoch,"loss",loss.item())


    loss_epochwise = [loss/(max(loss_epochwise)) for loss in loss_epochwise]
    #plotting loss with masking=85% and batch_size=32
    a= plt.plot([i+1 for i in range(10)], loss_epochwise)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Epochwise')

    path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/saved_mae_models/mae_trial.pt"
    path2= "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/saved_mae_models/mae_trial_plt.jpg"
    plt.savefig(path2)
    torch.save(mae, path)

    image_400 = dataset.__getitem__(400)
    image_400 = torch.reshape(image_400,(1,2,256,256))
    image_patch = rearrange(image_400,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h = 8, w = 8, p1 = 32, p2 = 32, c = 2)

    unmasked_img = np.zeros((1,64,2048))
    
    mae.eval()
    with torch.no_grad():
        loss = mae(image_400.to(device))
        re_image_400 = mae.final()
        latent = mae.repre_latent()
        unmasked_ind = mae.unmasked_ind()
        