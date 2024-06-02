r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput
import numpy as np

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, stage,dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        # self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(3, 1)
        self.dropout = build_dropout_layer(dropout)
        self.stage = stage

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        windows_attent_filg = False

        if(self.stage==0):
            win_w = 2
            win_h = 2
            windows_attent_filg = True
        elif(self.stage==1):
            win_w = 4
            win_h = 4
            windows_attent_filg = True

        if(windows_attent_filg == True):
            add_row_num = int(np.ceil(input_q.shape[1] / (win_w * win_h)) * win_w * win_h) - input_q.shape[1]
            q = rearrange(torch.cat([self.proj_q(input_q), torch.zeros((1, add_row_num, self.d_model), dtype=torch.float32).cuda()], 1)
                          , 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange( torch.cat([self.proj_k(input_k), torch.zeros((1, add_row_num, self.d_model), dtype=torch.float32).cuda()], 1)
                          , 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange( torch.cat([self.proj_v(input_v), torch.zeros((1, add_row_num, self.d_model), dtype=torch.float32).cuda()], 1)
                          , 'b m (h c) -> b h m c', h=self.num_heads)

            proj = self.proj_p(embed_qk[self.stage])
            p = proj.repeat(1, 1, 1, self.num_heads).permute(0, 3, 1, 2)

            q = q.view(q.shape[0], self.num_heads, q.shape[2] // (win_w*win_h),(win_w*win_h), q.shape[3]).transpose(1, 2)
            q = q.view(q.shape[0]*q.shape[1],self.num_heads,q.shape[-2],q.shape[-1])

            k = k.view(k.shape[0], self.num_heads, k.shape[2] // (win_w*win_h),(win_w*win_h), k.shape[3]).transpose(1, 2)
            k = k.view(k.shape[0]*k.shape[1],self.num_heads,k.shape[-2],k.shape[-1])

            v = v.view(v.shape[0], self.num_heads, v.shape[2] // (win_w*win_h), (win_w * win_h), v.shape[3]).transpose(1, 2)
            v = v.view(v.shape[0] * v.shape[1], self.num_heads, v.shape[-2], v.shape[-1])

            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
            attention_scores = (attention_scores_e + p) / self.d_model_per_head ** 0.5
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_scores = self.dropout(attention_scores)

            hidden_states = torch.matmul(attention_scores, v)
            first = hidden_states.shape[0]
            second = hidden_states.shape[1]
            third = hidden_states.shape[2]
            fourth = hidden_states.shape[3]


            hidden_states = hidden_states.view(first * second * third, fourth).view(1, second, first * third, fourth)
            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
            return hidden_states[:, 0:hidden_states.shape[1] - add_row_num, :], attention_scores

        else:
            q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)


            # p = rearrange(self.proj_p(embed_qk[self.stage]), 'b n m (h c) -> b h n m c', h=self.num_heads)

            proj = self.proj_p(embed_qk[self.stage])

            p = proj.repeat(1, 1, 1, self.num_heads).permute(0, 3, 1, 2)

            # attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)
            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
            attention_scores = (attention_scores_e + p) / self.d_model_per_head ** 0.5
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_scores = self.dropout(attention_scores)

            hidden_states = torch.matmul(attention_scores, v)

            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
            return hidden_states, attention_scores

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def create_mask(H,W,win_h,win_w,window_size,shift_size):
    Hp = int(np.ceil(H / win_h)) * window_size
    Wp = int(np.ceil(W / win_w)) * window_size
    img_mask = torch.zeros((1, Hp, Wp, 1)).cuda()  # 1 Hp Wp 1
    h_slices = (slice(0, -1*window_size),
                slice(-1*window_size, -1*shift_size),
                slice(-1*shift_size, None))
    w_slices = (slice(0, -1*window_size),
                slice(-1*window_size, -1*shift_size),
                slice(-1*shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class SWIFT_RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, stage,dropout=None):
        super(SWIFT_RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        # self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(3, 1)

        self.dropout = build_dropout_layer(dropout)
        self.stage = stage

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        windows_attent_filg = False

        if(self.stage==0):
            win_w = 2
            win_h = 2
            mask = torch.zeros((win_w*win_h,win_w*win_h)).cuda()

            mask_len = win_w*win_h//2

            mask[0:mask_len,mask_len:] = -100
            mask[mask_len:,0:mask_len] = -100
            windows_attent_filg = True
        elif(self.stage==1):
            win_w = 4
            win_h = 4

            mask_len = win_w * win_h // 2

            mask = torch.zeros((win_w*win_h, win_w*win_h)).cuda()
            mask[0:mask_len,mask_len:] = -100
            mask[mask_len:,0:mask_len] = -100
            windows_attent_filg = True

        if(windows_attent_filg == True):

            add_row_num = int(np.ceil(input_q.shape[1] / (win_w * win_h)) * win_w * win_h) - input_q.shape[1]

            input_q_cat = torch.cat([input_q, torch.zeros((1, add_row_num, self.d_model), dtype=torch.float32).cuda()], 1)
            input_q_cat_roll = torch.zeros((input_q_cat.shape[0],input_q_cat.shape[1],input_q_cat.shape[2]), dtype=torch.float32).cuda()
            input_q_cat_roll[:,0:input_q_cat.shape[1]-(win_w * win_h)//2,:]  = input_q_cat[:,((win_w * win_h)//2):,:]
            input_q_cat_roll[:,input_q_cat.shape[1] - (win_w * win_h)//2:,:] = input_q_cat[:,0:(win_w * win_h)//2,:]

            q = rearrange(self.proj_q(input_q_cat_roll)
                          , 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_q_cat_roll)
                          , 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange(self.proj_k(input_q_cat_roll)
                          , 'b m (h c) -> b h m c', h=self.num_heads)

            proj = self.proj_p(embed_qk[self.stage])
            p = proj.repeat(1, 1, 1, self.num_heads).permute(0, 3, 1, 2)

         
            q = q.view(q.shape[0], self.num_heads, q.shape[2] // (win_w*win_h),(win_w*win_h), q.shape[3]).transpose(1, 2)
            q = q.view(q.shape[0]*q.shape[1],self.num_heads,q.shape[-2],q.shape[-1])

            k = k.view(k.shape[0], self.num_heads, k.shape[2] // (win_w*win_h),(win_w*win_h), k.shape[3]).transpose(1, 2)
            k = k.view(k.shape[0]*k.shape[1],self.num_heads,k.shape[-2],k.shape[-1])

            v = v.view(v.shape[0], self.num_heads, v.shape[2] // (win_w*win_h), (win_w * win_h), v.shape[3]).transpose(1, 2)
            v = v.view(v.shape[0] * v.shape[1], self.num_heads, v.shape[-2], v.shape[-1])

            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
            attention_scores = (attention_scores_e + p) / self.d_model_per_head ** 0.5

            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = attention_scores + mask
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_scores = self.dropout(attention_scores)

            hidden_states = torch.matmul(attention_scores, v)
            first = hidden_states.shape[0]
            second = hidden_states.shape[1]
            third = hidden_states.shape[2]
            fourth = hidden_states.shape[3]


            hidden_states = hidden_states.view(first * second * third, fourth).view(1, second, first * third, fourth)
            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

            hidden_states_reverse = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
                                           dtype=torch.float32).cuda()

            hidden_states_reverse[:, 0:(win_w * win_h) // 2, :] = hidden_states[:,hidden_states.shape[1]-((win_w * win_h) // 2):, :]
            hidden_states_reverse[:,(win_w * win_h) // 2:,:] = hidden_states[:,0:hidden_states.shape[1]-((win_w * win_h) // 2),:]

            return hidden_states[:, 0:hidden_states.shape[1] - add_row_num, :], attention_scores

        else:
            q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

            proj = self.proj_p(embed_qk[self.stage])
            p = proj.repeat(1, 1, 1, self.num_heads).permute(0, 3, 1, 2)

            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
            attention_scores = (attention_scores_e + p) / self.d_model_per_head ** 0.5
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_scores = self.dropout(attention_scores)

            hidden_states = torch.matmul(attention_scores, v)

            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
            return hidden_states, attention_scores


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads,stage, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads,stage, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class swift_RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads,stage, dropout=None):
        super(swift_RPEAttentionLayer, self).__init__()
        self.swift_attention = SWIFT_RPEMultiHeadAttention(d_model, num_heads,stage, dropout=dropout)
        self.swift_linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.swift_attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.swift_linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores

class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads,stage, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads,stage, dropout=dropout)
        self.swift_attention = swift_RPEAttentionLayer(d_model, num_heads,stage, dropout=dropout)

        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )

        hidden_states, attention_scores = self.swift_attention(
            hidden_states,
            hidden_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )

        output_states = self.output(hidden_states)
        return output_states, attention_scores
