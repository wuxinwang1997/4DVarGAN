# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial, lru_cache
import numpy as np
import torch
import torch.nn as nn
from src.models.forecast.layers.fno_layers import Block
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from src.utils.model_utils import load_constant
from src.models.forecast.layers.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed
)

class AFNONet(nn.Module):
    """Implements the AFNONet model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        num_blocks (int): number of fno blocks
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        double_skip (bool): whether to use residual twice
    """

    def __init__(
        self,
        img_size=[64, 128],
        patch_size=4,
        embed_dim=768,
        depth=8,
        num_blocks=8,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        double_skip=True,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = 1
        self.h = self.img_size[0] // patch_size
        self.w = self.img_size[1] // patch_size
        self.embed_dim = embed_dim

        # variable tokenization: separate embedding layer for each input variable
        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim,
                      mlp_ratio=mlp_ratio,
                      drop=drop_rate,
                      drop_path=dpr[i],
                      norm_layer=norm_layer,
                      double_skip=double_skip,
                      num_blocks=num_blocks,
                      sparsity_threshold=sparsity_threshold,
                      hard_thresholding_fraction=hard_thresholding_fraction)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.Linear(embed_dim, patch_size ** 2)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.patch_size, self.patch_size, self.c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.c, self.h * self.patch_size, self.w * self.patch_size))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
        x = self.patch_embed(x)

        # add pos embedding
        x = x + self.pos_embed

        x = self.pos_drop(x)

        x = x.reshape(-1, self.h, self.w, self.embed_dim)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)

        return preds
