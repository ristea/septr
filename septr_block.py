# This code is released under the CC BY-SA 4.0 license.

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Scale(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return x * self.val


class SepTrBlock(nn.Module):
    def __init__(self, channels, input_size, heads=3, mlp_dim=128, dim_head=32,
                 down_sample_input=None, project=False, reconstruct=False, dim=128, dropout_tr=0.0):
        super().__init__()
        patch_height, patch_width = pair(input_size)
        self.avg_pool = nn.Identity()
        self.upsample = nn.Identity()
        self.projection = nn.Identity()
        self.reconstruction = nn.Identity()

        if down_sample_input is not None:
            patch_height = patch_height // down_sample_input[0]
            patch_width = patch_width // down_sample_input[1]

            self.avg_pool = nn.AvgPool2d(kernel_size=down_sample_input)
            self.upsample = nn.UpsamplingNearest2d(scale_factor=down_sample_input)

        if project:
            self.projection = nn.Linear(channels, dim)
        if reconstruct:
            self.reconstruction = nn.Sequential(
                nn.Linear(dim, channels),
                Scale(dim ** -0.5)
            )

        self.rearrange_patches_h = Rearrange('b c h w -> b w h c')
        self.rearrange_patches_w = Rearrange('b c h w -> b h w c')

        self.rearrange_in_tr = Rearrange('b c h w -> (b c) h w')
        self.rearrange_out_tr_h = Rearrange('(b c) h w -> b w h c', c=patch_width)
        self.rearrange_out_tr_w = Rearrange('(b c) h w -> b w c h', c=patch_height)

        self.pos_embedding_w = nn.Parameter(torch.randn(1, 1, patch_width + 1, dim))
        self.pos_embedding_h = nn.Parameter(torch.randn(1, 1, patch_height + 1, dim))
        self.transformer_w = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout_tr)
        self.transformer_h = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout_tr)

    def forward(self, x, cls_token):
        x = self.avg_pool(x)

        # H inference
        h = self.rearrange_patches_h(x)
        h = self.projection(h)

        dim1, dim2, _, _ = h.shape
        if cls_token.shape[0] == 1:
            cls_token = repeat(cls_token, '() () n d -> b w n d', b=dim1, w=dim2)
        else:
            cls_token = repeat(cls_token, 'b () n d -> b w n d', w=dim2)

        h = torch.cat((cls_token, h), dim=2)
        h += self.pos_embedding_h

        h = self.rearrange_in_tr(h)
        h = self.transformer_h(h)
        h = self.rearrange_out_tr_h(h)

        # W inference
        w = self.rearrange_patches_w(h[:, :, 1:, :])

        cls_token = h[:, :, 0, :].unsqueeze(2)
        cls_token = repeat(cls_token.mean((-1, -2)).unsqueeze(1).unsqueeze(1), 'b () d2 e -> b d1 d2 e', d1=w.shape[1])

        w = torch.cat((cls_token, w), dim=2)
        w += self.pos_embedding_w

        w = self.rearrange_in_tr(w)
        w = self.transformer_w(w)
        w = self.rearrange_out_tr_w(w)

        x = self.upsample(w[:, :, :, 1:])
        x = self.reconstruction(x)

        cls_token = w[:, :, :, 0].mean(2).unsqueeze(1).unsqueeze(1)
        return x, cls_token
