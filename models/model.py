from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.layers import trunc_normal_
from ViT_model import VisionTransformer
from utils import compute_rotation_matrix_from_ortho6d
import matplotlib.pyplot as plt
import seaborn as sns
MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
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

# IMPORTANT, WE HAD TO CHANGE THE DEFAULT NUM OF HEADS FROM 12 TO 8, OTHERWISE THE MODEL WOULD NOT LOAD THE PRETRAINED WEIGHTS (BECAUSE THE PRETRAINED WEIGHTS WERE TRAINED WITH 8 HEADS)
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., num_ori_tokens=None, scale_with_head=False, show_attns=False,n_dep=0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.show_attns = show_attns
        self.num_ori_tokens = num_ori_tokens
        self.n_dep = n_dep

    def plot_attention(self, attn, type="single head", head_index=5):
        if not os.path.exists('output/vis'):
            os.makedirs('output/vis')
        if type == "single head":
            values = attn[0,head_index,0:self.num_ori_tokens,0:self.num_ori_tokens].detach().cpu()
        else: # all heads
            values = torch.sum(attn,dim=1)
            values = values[0, 0:self.num_ori_tokens, 0:self.num_ori_tokens].detach().cpu()

        fig = plt.figure()
        sns.heatmap(values, cmap='plasma')
        fig.savefig(f"./output/vis/attention interaction in layer {self.n_dep+1}.png", bbox_inches='tight')
        plt.show()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        if self.show_attns == True:
            self.plot_attention(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_ori_tokens=None,
                 all_attn=False, scale_with_head=False, show_attns=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_ori_tokens = num_ori_tokens
        for n_dep in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_ori_tokens=num_ori_tokens,
                                                scale_with_head=scale_with_head,show_attns=show_attns,n_dep=n_dep))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None, pos=None):
        for idx, (attn, ff) in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_ori_tokens:] += pos
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Orientation_Blocks(nn.Module):
    """
    feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
    """
    def __init__(self, *, num_ori_tokens, dim, depth, heads, mlp_dim,
                 dropout=0., emb_dropout=0., pos_embedding_type="learnable",
                 ViT_feature_dim, ViT_feature_num, w, h, inference_view=False):
        """
        inference_view: In inference stage, for a single image input, show the ori_tokens similarity matrix
        and the attention interactions of the learned ori_tokens in each Transformer layer.
        """
        super().__init__()
        patch_dim = ViT_feature_dim
        self.inplanes = 64
        self.num_ori_tokens = num_ori_tokens
        self.num_patches = ViT_feature_num
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.ori_tokens = nn.Parameter(torch.zeros(1, self.num_ori_tokens, dim))

        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.inference_view = inference_view

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, num_ori_tokens=num_ori_tokens,
                                       all_attn=self.all_attn, show_attns=self.inference_view)

        self.to_ori_token = nn.Identity()

        self.to_dir_6_d = nn.Sequential(
            nn.Linear(dim, 6)
        )

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_ori_tokens, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def plot_sim_matrix(self, A, type="cos-similarity"):
        """
        visualize the similarity matrix of orientation tokens
        type: "softmax/cos-similarity"
        """
        assert type == "softmax" or type == "cos-similarity", "please use correct type. (softmax/cos-similarity)"

        if not os.path.exists('output/vis'):
            os.makedirs('output/vis')

        if type=="softmax":
            in_pro = A.mm(A.T) / math.sqrt(A.shape[1])
            prob = F.softmax(in_pro, dim=0)
            fig = plt.figure()
            sns.heatmap(prob, cmap='plasma')
            fig.savefig("./output/vis/softmax_similarity_matrix_of_ori_tokens.png", bbox_inches='tight')
            plt.show()
        elif type=="cos-similarity":
            a = (A / torch.norm(A, dim=-1, keepdim=True) )[0,...]
            similarity = torch.mm(a, a.T)
            fig = plt.figure()
            sns.heatmap(similarity, cmap='plasma') # options: YlGnBu plasma
            fig.savefig("./output/vis/cos_similarity_matrix_of_ori_tokens.png", bbox_inches='tight')
            plt.show()

    def forward(self, features, mask=None):
        """
        feature extractor (ViT) -> add ori_tokens -> Orientation_Blocks -> outputs in all regions
        """
        # show ori_tokens similarity matrix
        if self.inference_view == True:
            self.plot_sim_matrix(self.ori_tokens.cpu())

        # transformer features
        x = self.patch_to_embedding(features) # shape [batch_size, channel=197, dim=192]
        b, n, _ = x.shape

        # add learnable orientation tokens
        ori_tokens = repeat(self.ori_tokens, '() n d -> b n d', b=b)

        # add pos_embedding
        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((ori_tokens, x), dim=1)
        elif self.pos_embedding_type == "learnable":
            x = torch.cat((ori_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_ori_tokens)]

        x = self.dropout(x)

        # feature extractor (ViT) -> Orientation_Blocks
        x = self.transformer(x, mask, self.pos_embedding)

        # Orientation_Blocks -> outputs in all regions
        dir_tokens = self.to_ori_token(x[:, 0:self.num_ori_tokens])
        dir_6_d = self.to_dir_6_d(dir_tokens)

        # convert to rotation matrices
        batch_size, num_ori_tokens, d = dir_6_d.size()
        x_reshaped = dir_6_d.view(-1, d)
        ori_9_d = compute_rotation_matrix_from_ortho6d(x_reshaped)

        # reshape to [batch_size, num_ori_tokens, 3d, 3d]
        ori_9_d = ori_9_d.view(batch_size, num_ori_tokens, 3,3)

        return ori_9_d



class TokenHPE(nn.Module):

    def __init__(self, num_ori_tokens=9,
                 depth=12, heads=8, embedding='sine-full', ViT_weights='',
                 dim=128, mlp_ratio=3, inference_view=False
                 ):
        super(TokenHPE, self).__init__()

        # Feature extractor (ViT)
        # VisionTransformer implemented by rwightman:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # use vit_base_patch16_224_in21k
        self.feature_extractor = VisionTransformer(
                              img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              mlp_head=False,
                              )

        # whether to use intermediate weights
        if ViT_weights != "":
            assert os.path.exists(ViT_weights), "weights file: '{}' not exist.".format(ViT_weights)
            weights_dict = torch.load(ViT_weights, map_location="cuda")
            # delete cls head weights
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print("use pretrained feature extractor (ViT) weights!")
            print(self.feature_extractor.load_state_dict(weights_dict, strict=False))

        # Transformer blocks with orientation tokens
        self.Ori_blocks = Orientation_Blocks(
                                         num_ori_tokens=num_ori_tokens,
                                         dim=dim,
                                         ViT_feature_dim=768,
                                         ViT_feature_num=197,
                                         w=14,
                                         h=14,
                                         depth=depth,
                                         heads=heads,
                                         mlp_dim=dim * mlp_ratio,
                                         pos_embedding_type=embedding,
                                         inference_view=inference_view
                                         )

        self.mlp_head = nn.Sequential(
            nn.Linear(num_ori_tokens*9, num_ori_tokens*27),
            nn.Tanh(),
            nn.Linear(num_ori_tokens*27, 6)
        )

    def forward(self, x):
        """
        TokenHPE pipeline
        feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
        -> MLP head -> prediction: [pred, ori_9_d]
        """
        # feature extractor (ViT)
        x = self.feature_extractor(x) # outputs: [batch_size, channel=197, dim = 768]

        # Orientation_Blocks
        ori_9_d = self.Ori_blocks(x) # [batch_size, num_ori_tokens, 3d, 3d]

        # feed to mlp head
        x = rearrange(ori_9_d, 'batch oris d_1 d_2-> batch (oris d_1 d_2)')

        x = self.mlp_head(x)

        pred = compute_rotation_matrix_from_ortho6d(x)

        return pred, ori_9_d


