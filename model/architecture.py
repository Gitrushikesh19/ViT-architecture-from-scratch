import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np

#Self-Attention
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.emb_dim = config['emb_dim']
        self.drop_prob = config['dropout'] if 'dropout' in config else 0.0
        self.att_dim = self.n_heads * self.head_dim
        
        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.att_dim, bias=False)
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.emb_dim),
            nn.Dropout(self.drop_prob))

        self.attn_dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        
        #  Converting to Attention Dimension
       
        # Batch Size x Number of Patches x Dimension
        B, N = x.shape[:2]
        # Projecting to 3*att_dim and then splitting to get q, k v(each of att_dim)
        # qkv -> Batch Size x Number of Patches x (3* Attention Dimension)
        # q(as well as k and v) -> Batch Size x Number of Patches x Attention Dimension
        q, k ,v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        # Batch Size x Number of Patches x Attention Dimension
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Heads x Number of Patches x Head Dimension
        # -> B x H x N x Head Dimension
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        
        # Compute Attention Weights

        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**(-0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted Value Computation

        #  B x H x N x N @ B x H x N x Head Dimension
        # -> B x H x N x Head Dimension
        out = torch.matmul(att, v)
             
        # Converting to Transformer Dimension
        
        # B x N x (Heads * Head Dimension) -> B x N x (Attention Dimension)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')
        #  B x N x Dimension
        out = self.output_proj(out)
        
        return out

#TransformerLayer
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config['emb_dim']
        ff_hidden_dim = config['ff_dim'] if 'ff_dim' in config else 4*emb_dim
        ff_drop_prob = config['ff_drop'] if 'ff_drop' in config else 0.0
        self.att_norm = nn.LayerNorm(emb_dim)
        self.attn_block = Attention(config)
        self.ff_norm = nn.LayerNorm(emb_dim)
        
        self.ff_block = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(ff_drop_prob),
            nn.Linear(ff_hidden_dim, emb_dim),
            nn.Dropout(ff_drop_prob)
        )
        
    def forward(self, x):
        out = x
        out = out + self.attn_block(self.att_norm(out))
        out = out + self.ff_block(self.ff_norm(out))
        return out

#Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Example configuration
        #   Image c,h,w : 3, 224, 224
        #   Patch h,w : 16, 16
        image_height = config['image_height']
        image_width = config['image_width']
        im_channels = config['im_channels']
        emb_dim = config['emb_dim']
        patch_embd_drop = config['patch_emb_drop']
        
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']
        
        # Compute number of patches for positional parameters initialization
        #   num_patches = num_patches_h * num_patches_w
        #   num_patches = 224/16 * 224/16
        #   num_patches = 196
        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        
        # This is the input dimension of the patch_embed layer
        # After patchifying the 224, 224, 3 image will be
        # num_patches x patch_h x patch_w x 3
        # Which will be 196 x 16 x 16 x 3
        # Hence patch dimension = 16 * 16 * 3
        patch_dim = im_channels * self.patch_height * self.patch_width
        
        self.patch_embed = nn.Sequential(
            # This pre and post layer norm speeds up convergence
            # Comment them if you want pure vit implementation
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        # Positional information needs to be added to cls as well so 1+num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(emb_dim))
        self.patch_emb_dropout = nn.Dropout(patch_embd_drop)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # This is doing the B, 3, 224, 224 -> (B, num_patches, patch_dim) transformation
        # B, 3, 224, 224 -> B, 3, 14*16, 14*16
        # B, 3, 14*16, 14*16 -> B, 3, 14, 16, 14, 16
        # B, 3, 14, 16, 14, 16 -> B, 14, 14, 16, 16, 3
        #  B, 14*14, 16*16*3 - > B, num_patches, patch_dim
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                      ph=self.patch_height,
                      pw=self.patch_width)
        out = self.patch_embed(out)
        
        # Add cls
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        out = torch.cat((cls_tokens, out), dim=1)
        
        # Add position embedding and do dropout
        out += self.pos_embed
        out = self.patch_emb_dropout(out)
        
        return out

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = config['n_layers']
        emb_dim = config['emb_dim']
        num_classes = config['num_classes']
        self.patch_embed_layer = PatchEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.fc_number = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        # Patchify and add CLS token
        out = self.patch_embed_layer(x)
        
        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        
        # Compute logits
        return self.fc_number(out[:, 0])
