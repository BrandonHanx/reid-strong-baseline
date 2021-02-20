import math
from functools import partial

import timm.models.vision_transformer as ViTcls
import torch
import torch.nn.functional as F
from timm.models.helpers import load_pretrained


class ViT(ViTcls.VisionTransformer):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.out_channels = self.embed_dim

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.mode == "first":
            return x[:, 0]
        if self.mode == "average":
            return x[:, 1:].mean(dim=1)
        return NotImplementedError

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if "fc" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def resize_pos_embed(posemb, posemb_new, gs_new):
    # Rescale the grid of position embeddings when loading from state_dict.
    print("Resized position embedding: {} to {}".format(posemb.shape, posemb_new.shape))
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bilinear", align_corners=True
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, gs_new):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed, gs_new)
        out_dict[k] = v
    return out_dict


def _create_vit(variant, mode, img_size, pretrained, patch_size, **kwargs):
    model = ViT(mode=mode, img_size=img_size, **kwargs)
    model.default_cfg = ViTcls.default_cfgs[variant]
    gs_new = (int(img_size[0] / patch_size), int(img_size[1] / patch_size))

    if pretrained:
        load_pretrained(
            model, filter_fn=partial(checkpoint_filter_fn, model=model, gs_new=gs_new)
        )
    return model


model_archs = {}
model_archs["vit_deit_small_patch16_224"] = dict(
    patch_size=16, embed_dim=384, depth=12, num_heads=6
)
model_archs["vit_deit_base_patch16_224"] = dict(
    patch_size=16, embed_dim=768, depth=12, num_heads=12
)


def deit(arch="vit_deit_small_patch16_224"):
    model_arch = model_archs[arch]
    return _create_vit(
        variant=arch, mode="first", img_size=(256, 128), pretrained=True, **model_arch
    )
