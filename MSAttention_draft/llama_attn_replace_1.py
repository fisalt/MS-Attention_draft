# Modified based on https://github.com/lm-sys/FastChat
import sys
import warnings
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
from torch import nn, einsum
import transformers
from einops import rearrange, repeat, pack, unpack
# from flash_attn import __version__ as flash_attn_version
# from flash_attn import "2.1.0" as flash_attn_version
flash_attn_version="2.1.0"
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_func
)

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, LlamaAttention
from flash_attn.bert_padding import unpad_input, pad_input
import math
from transformers.models.llama.configuration_llama import LlamaConfig
from local_attention import LocalAttention


group_size_ratio = 1/16
def forward_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if not self.training:
        warnings.warn("This function should be used just for training as it may exhibit reduced inference performance. For inference, please use forward_flashattn_inference.")

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    # key_padding_mask = attention_mask.repeat(2, 1, 1, 1)
    # print("attention_mask:",attention_mask.shape)
    # print("attention_mask:",attention_mask)
    # key_padding_mask = attention_mask[0,0,:,0][None,:].repeat(2,1)
    # key_padding_mask = attention_mask[0,0,:,0][None,:]
    nheads = qkv.shape[-2]
    # shift

    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))

    # qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(bsz * 2,
    #                                                                                                           q_len, 3,
    #                                                                                                           self.num_heads // 2,
    #                                                                                                           self.head_dim)
    # x = rearrange(qkv, "b s three h d -> b s (three h d)")
    # x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    # cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
    # cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + group_size // 2]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
    # cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)

    # x_unpad = rearrange(
    #     x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    # )
    # output_unpad = flash_attn_varlen_qkvpacked_func(
    #     x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    # )
    # output = rearrange(
    #     pad_input(
    #         rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
    #     ),
    #     "b s (h d) -> b s h d",
    #     h=nheads // 2,
    # )
    output = flash_attn_func(
                query_states, key_states, value_states, softmax_scale=0.0, causal=True
            )
    # print(output.shape)
    output=output.transpose(1, 2).reshape(bsz, q_len, nheads, self.head_dim)
    # output = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(bsz, q_len, nheads,
    #                                                                                            self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = padding_mask
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift
    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv

    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )

    # attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    # if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    local_mask=torch.ones((group_size,group_size),dtype=torch.int).cuda()
    local_mask=torch.tril(local_mask, diagonal=0)
    local_mask = local_mask[None, None, :group_size, :group_size].repeat(bsz * num_group, self.num_heads, 1, 1).bool()
    mask_value=-torch.finfo(key_states.dtype).max
    attn_weights.masked_fill_(~local_mask, mask_value)
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # shift back
    attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def forward_noflashattn1(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift
    # def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
    #     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
    #     qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
    #     return qkv

    # query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    query_r=query_states[:,self.num_heads//2:]
    key_r=key_states[:,self.num_heads//2:]
    value_r=value_states[:,self.num_heads//2:]

    query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
    key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
    value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
    query_l=query_states[:,:self.num_heads//2]
    key_l=key_states[:,:self.num_heads//2]
    value_l=value_states[:,:self.num_heads//2]
    attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
    #         f" {attn_weights.size()}"
    #     )

    # attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    # if attention_mask is not None:
    #     if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights
        # attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    local_mask=torch.ones((group_size,group_size),dtype=torch.int).cuda()
    local_mask=torch.tril(local_mask, diagonal=0)
    local_mask = local_mask[None, None, :group_size, :group_size].repeat(bsz * num_group, self.num_heads//2, 1, 1).bool()
    mask_value=-torch.finfo(key_states.dtype).max
    attn_weights.masked_fill_(~local_mask, mask_value)
    # local_mask=torch.ones((num_group,2*num_group),dtype=torch.int).cuda()
    # local_mask=torch.tril(local_mask, diagonal=num_group)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_l)

    # if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

    attn_output2,attn_weights2=forward_mss_inline(self,q=query_r,k=key_r,v=value_r)
    # print(attn_output.shape,attn_output2.shape)

    attn_output=torch.cat((attn_output,attn_output2),dim=-2)
    # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

    # shift back
    # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def forward_noflashattn2(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift
    # def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
    #     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
    #     qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
    #     return qkv

    # query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    query_r=query_states[:,self.num_heads//2:]
    key_r=key_states[:,self.num_heads//2:]
    value_r=value_states[:,self.num_heads//2:]

    # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
    # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
    query_l=query_states[:,:self.num_heads//2]
    key_l=key_states[:,:self.num_heads//2]
    value_l=value_states[:,:self.num_heads//2]


    # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

    attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

    attn_output2,attn_weights=forward_mss_inline(self,q=query_r,k=key_r,v=value_r)
    # print(attn_output.shape,attn_output2.shape)

    attn_output=torch.cat((attn_output,attn_output2),dim=-2)
    # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

    # shift back
    # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value

def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def forward_localw_inline(
    self,
    q, k, v,
    window_size = 128
):

    # mask = default(mask, input_mask)

    # assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

    shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, True, -1, window_size, True, 1, 0, False

    # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
    # (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
    (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

    # auto padding

    if autopad:
        orig_seq_len = q.shape[1]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, window_size, dim = -2), (q, k, v))

    b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

    scale = dim_head ** -0.5

    assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

    windows = n // window_size

    seq = torch.arange(n, device = device)
    b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

    # bucketing

    bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

    bq = bq * scale

    look_around_kwargs = dict(
        backward =  look_backward,
        forward =  look_forward,
        pad_value = pad_value
    )

    bk = look_around1(bk, **look_around_kwargs)
    bv = look_around1(bv, **look_around_kwargs)

    # rotary embeddings

    # if exists(self.rel_pos):
    #     pos_emb, xpos_scale = self.rel_pos(bk)
    #     bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

    # calculate positions for masking

    bq_t = b_t
    bq_k = look_around1(b_t, **look_around_kwargs)

    bq_t = rearrange(bq_t, '... i -> ... i 1')
    bq_k = rearrange(bq_k, '... j -> ... 1 j')

    pad_mask = bq_k == pad_value

    sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

    # if exists(attn_bias):
    #     heads = attn_bias.shape[0]
    #     assert (b % heads) == 0

    #     attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
    #     sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    # if shared_qk:
    #     self_mask = bq_t == bq_k
    #     sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
    #     del self_mask

    if causal:
        causal_mask = bq_t < bq_k

        # if self.exact_windowsize:
        #     max_causal_window_size = (self.window_size * self.look_backward)
        #     causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

        sim = sim.masked_fill(causal_mask, mask_value)
        del causal_mask

    # masking out for exact window size for non-causal
    # as well as masking out for padding value

    if not causal: #and self.exact_windowsize:
        max_backward_window_size = (window_size * look_backward)
        max_forward_window_size = (window_size * look_forward)
        window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
        sim = sim.masked_fill(window_mask, mask_value)
    else:
        sim = sim.masked_fill(pad_mask, mask_value)

    # take care of key padding mask passed in

    # if exists(mask):
    #     batch = mask.shape[0]
    #     assert (b % batch) == 0

    #     h = b // mask.shape[0]

    #     if autopad:
    #         _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

    #     mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
    #     mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
    #     mask = rearrange(mask, '... j -> ... 1 j')
    #     mask = repeat(mask, 'b ... -> (b h) ...', h = h)
    #     sim = sim.masked_fill(~mask, mask_value)
    #     del mask

    # attention

    attn = sim.softmax(dim = -1)
    # attn = self.dropout(attn)

    # aggregation

    out = einsum('b h i j, b h j e -> b h i e', attn, bv)
    out = rearrange(out, 'b w n d -> b (w n) d')

    if autopad:
        out = out[:, :orig_seq_len, :]

    out, *_ = unpack(out, packed_shape, '* n d')
    return out

def forward_mss_inline(self,q,k,v):
    # bsz, q_len, _ = q.size()
    topklist=[256,196,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*num_heads,n1,device='cuda')
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = rearrange(attn_output, 'b n h d -> b n (h d)')

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

    return attn_output, attn_weights

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)


def forward_mss_inline_own(self,q,k,v):
    # bsz, q_len, _ = q.size()
    topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*num_heads,n1,device='cuda')
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots[...,:sel0]+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dots[...,sel0:sel0+sel1]+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)+1), dots[...,sel0+sel1:]+torch.log((self.patchscale[2])%(s)+1)), dim=-1)
    dots=torch.cat((dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = rearrange(attn_output, 'b n h d -> b n (h d)')

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

    return attn_output, attn_weights

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)


def replace_llama_attn(use_flash_attn=True, use_full=False, inference=False):
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_inference
        else:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
                _prepare_decoder_attention_mask
            )
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_full if use_full else forward_flashattn
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noflashattn
        # transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention_mss
        # sys.modules['transformers.models.llama.modeling_llama.LlamaAttention'] = LlamaAttention
        # from importlib import reload
        # reload(sys.modules['transformers.models.llama.modeling_llama.LlamaAttention'])
        # sys.modules['transformers.models.llama.modeling_llama'].LlamaAttention = LlamaAttention_mss
        # transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention_mss
        # transformers.models.llama.modeling_llama.LlamaDecoderLayer.self_attn = LlamaAttention_mss
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_mss


def forward_mss_inline_own2(self,q,k,v,topklist=[300,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    # if n*s<=sel0+sel1:
    v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    kv_mask=selectgs
    g_mask=torch.arange(0,n,1,device='cuda')

    mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
    maskg = (q_mask[:, :, None]//s) > g_mask[:]
    maskg=maskg[None, None, :, :]

    mask_value=-torch.finfo(k.dtype).max
    dots.masked_fill_(~mask, mask_value)
    dotsg.masked_fill_(~maskg, mask_value)
    # dotsl.masked_fill_(~maskl, mask_value)
    # dotsl.masked_fill_(causal_mask, mask_value)
    # dotsl = dotsl.masked_fill(pad_mask, mask_value)
    del mask
    del maskg

    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dots, dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn=nn.functional.softmax(dots, dim=-1)

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    out = out.transpose(1, 2).contiguous()

    return (out), attn


def forward_mss_inline_own1(self,q,k,v,topklist=[400,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    km=k
    vm=v
    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    # if n*s<=sel0+sel1:
    v=v0
    mask_value=-torch.finfo(k.dtype).max
    # v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    kv_mask=selectgs
    mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
    dots.masked_fill_(~mask, mask_value)
    del mask
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # q_mask=torch.arange(0,n*s,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # kv_mask=selectgs
    # g_mask=torch.arange(0,n,1,device='cuda')

    # mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
    # maskg = (q_mask[:, :, None]//s) > g_mask[:]
    # maskg=maskg[None, None, :, :]

    # mask_value=-torch.finfo(k.dtype).max
    # dots.masked_fill_(~mask, mask_value)
    # dotsg.masked_fill_(~maskg, mask_value)
    # dotsl.masked_fill_(~maskl, mask_value)
    # dotsl.masked_fill_(causal_mask, mask_value)
    # dotsl = dotsl.masked_fill(pad_mask, mask_value)
    # del mask
    # del maskg

    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots, dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn=nn.functional.softmax(dots, dim=-1)

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    out = out.transpose(1, 2).contiguous()

    return (out), attn

def forward_mss_inline_ownj1(self,q,k,v,topklist=[400,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    km=k
    vm=v
    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    _,_,n2,_=global_tokenk.shape
    dotsgqs=torch.matmul(global_tokenq[...,None,:],q.transpose(-1, -2))*scaling
    dotsgqs=nn.functional.softmax(dotsgqs, dim=-1)
    global_tokenq=dotsgqs@q

    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    # if n*s<=sel0+sel1:
    v=v0
    mask_value=-torch.finfo(k.dtype).max
    # v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    kv_mask=selectgs
    mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
    dots.masked_fill_(~mask, mask_value)
    del mask
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # q_mask=torch.arange(0,n*s,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # kv_mask=selectgs
    # g_mask=torch.arange(0,n,1,device='cuda')

    # mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
    # maskg = (q_mask[:, :, None]//s) > g_mask[:]
    # maskg=maskg[None, None, :, :]

    # mask_value=-torch.finfo(k.dtype).max
    # dots.masked_fill_(~mask, mask_value)
    # dotsg.masked_fill_(~maskg, mask_value)
    # dotsl.masked_fill_(~maskl, mask_value)
    # dotsl.masked_fill_(causal_mask, mask_value)
    # dotsl = dotsl.masked_fill(pad_mask, mask_value)
    # del mask
    # del maskg

    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots, dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn=nn.functional.softmax(dots, dim=-1)

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    out = out.transpose(1, 2).contiguous()

    return (out), attn


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def exists(val):
    return val is not None
def default(value, d):
    return d if not exists(value) else value
def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 3):
    t = x.shape[2]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:,:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

def look_around1(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

def extract_unique_elements(selectm, k):
    # 获取每个张量最后一维的unique元素
    # sorted_elements, _ = torch.sort(selectm, dim=-1)
    sorted_elements, sorted_indx = torch.sort(selectm, dim=-1)
    m=torch.max(sorted_indx)
    # print(sorted_indx.shape)
    # m=torch.max(sorted_indx, dim=-1)[0]
    # sorted_elements=torch.cat((sorted_elements,torch.zeros_like(sorted_elements[...,-1])[...,None]))
    sorted_elements2=torch.cat(((sorted_elements[...,-1])[...,None],sorted_elements),dim=-1)
    mask=torch.where(sorted_elements2[...,:-1]==sorted_elements2[...,1:])
    # sorted_elements[mask]=m
    sorted_indx[mask]=m
    # print(sorted_elements.shape)
    unique_elements, _ = torch.sort(sorted_indx, dim=-1)
    # unique_elements = torch.unique_consecutive(sorted_elements, dim=-1)
    # unique_elements,return_counts = torch.unique_consecutive(sorted_elements, dim=-1,return_counts=True)
    # print(unique_elements.shape)
    # print(sorted_elements.shape)
    # # unique_elements, inverse_indices,return_counts = torch.unique(selectm, dim=-1, return_inverse=True,return_counts=True,sorted=True)

    # # 对unique元素按值排序
    # # sorted_elements, _ = torch.sort(unique_elements, dim=-1)

    # # 提取每个张量最后一维的前k个元素
    # topk_elements = return_counts[..., :k]
    # result_tensor = torch.gather(sorted_elements, 0, topk_elements)
    topk_elements = unique_elements[..., :k]
    result_tensor = torch.gather(selectm, -1, topk_elements)


    return result_tensor


def extract_unique_elements1(selectm, k):
    # 获取每个张量最后一维的unique元素
    # sorted_elements, _ = torch.sort(selectm, dim=-1)
    sorted_elements, sorted_indx = torch.sort(selectm, dim=-1)
    # m=torch.max(sorted_indx)
    m=torch.max(sorted_indx, dim=-1)
    # sorted_elements=torch.cat((sorted_elements,torch.zeros_like(sorted_elements[...,-1])[...,None]))
    sorted_elements2=torch.cat(((sorted_elements[...,-1])[...,None],sorted_elements),dim=-1)
    mask=torch.where(sorted_elements2[...,:-1]==sorted_elements2[...,1:])
    # sorted_elements[mask]=m
    sorted_indx[mask]=m
    # print(sorted_elements.shape)
    unique_elements, _ = torch.sort(sorted_indx, dim=-1)
    # unique_elements = torch.unique_consecutive(sorted_elements, dim=-1)
    # unique_elements,return_counts = torch.unique_consecutive(sorted_elements, dim=-1,return_counts=True)
    # print(unique_elements.shape)
    # print(sorted_elements.shape)
    # # unique_elements, inverse_indices,return_counts = torch.unique(selectm, dim=-1, return_inverse=True,return_counts=True,sorted=True)

    # # 对unique元素按值排序
    # # sorted_elements, _ = torch.sort(unique_elements, dim=-1)

    # # 提取每个张量最后一维的前k个元素
    # topk_elements = return_counts[..., :k]
    # result_tensor = torch.gather(sorted_elements, 0, topk_elements)
    topk_elements = unique_elements[..., :k]
    result_tensor = torch.gather(selectm, -1, topk_elements)


    return result_tensor
class SinusoidalEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False
    ):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb1(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale ** -1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k

class MaSAd(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1, num_wins=7, topk=(64,32,32), patchsize=(1,4,0), patchscale=(1,2,10), causal_mask=True, num_mem_kv=1024, seg_len=128,max_seq_len=8192):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.to_qkv = nn.Linear(embed_dim, embed_dim*3, bias=True)
        # self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.seg_len=seg_len

        self.num_wins = num_wins
        # self.num_wins = 8
        self.topk = topk
        self.patchsize = patchsize
        self.patchscale = patchscale
        self.patchscale1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.attend=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(0.)
        self.causal_mask=causal_mask
        if patchscale[0]==0:
            self.patchscale=nn.ParameterList()
            # self.patchscale=[]
            self.patchscale.append(nn.Parameter(torch.tensor(1.0),requires_grad=True))
            self.patchscale.append(nn.Parameter(torch.tensor(4.0,requires_grad=True),requires_grad=True))
            self.patchscale.append(nn.Parameter(torch.tensor(10.0,requires_grad=True),requires_grad=True))
            # self.patchscale = nn.ParameterList([nn.Parameter(torch.tensor(1.0),requires_grad=True) for i in range(3)])
        # attnscale=torch.ones_like((1,1,1))


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        # self.num_mem_kv = max(num_mem_kv, 1 if causal_mask and not shared_qk else 0)
        # self.mem_key = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))
        # self.mem_value = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))
        xpos_scale_base=None
        self.rel_pos = SinusoidalEmbeddings(
                self.head_dim,
                use_xpos = False,
                scale_base = default(xpos_scale_base, seg_len // 2)
            )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.to_qkv.weight, gain=2 ** -2.5)
        # nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        # nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        # nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    

    def forward(self, q,k,v, autopad=True, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''

        # print(x.shape)
        # x=rearrange(x, 'b (n s) d -> b n s d',n=self.seg_len)
        
        # b,n,s,d=x.shape
        # b,seq,d=x.shape

        # q = self.q_proj(x)
        # k = self.k_proj(x)
        # v = self.v_proj(x)
        # if autopad:
        #     orig_seq_len = q.shape[1]
        #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.seg_len, dim = -2), (q, k, v))
        self.seg_len=64
        if autopad:
            orig_seq_len = q.shape[2]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, 64, dim = -2), (q, k, v))

        # x=rearrange(x, 'b (n s) d -> b n s d',s=self.seg_len)
        # self.seg_len=64
        # q=rearrange(q, 'b (n s) (h d) -> b h n s d',n=self.seg_len, h = self.num_heads)
        q=rearrange(q, 'b h (n s) d -> b h n s d',n=self.seg_len, h = self.num_heads)
        k=rearrange(k, 'b h (n s) d -> b h n s d',n=self.seg_len, h = self.num_heads)
        v=rearrange(v, 'b h (n s) d -> b h n s d',n=self.seg_len, h = self.num_heads)
        b,h,n,s,d=q.shape
        dots_shortcut=torch.sum(q*k*self.scaling,dim=-1)
        v_shortcut=v
        k_shortcut=k

        look_around_kwargs = dict(
            backward =  1,
            forward =  0,
            pad_value = -1
        )

        local_k = look_around(k, **look_around_kwargs)
        local_v = look_around(v, **look_around_kwargs)
        
        # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
        # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)
        pos_emb, xpos_scale = self.rel_pos(local_k)
        q, local_k = apply_rotary_pos_emb1(q, local_k, pos_emb, scale = xpos_scale)
        
        # bsz, h, w, _ = x.size()
        # mask_h, mask_w = rel_pos
        # mask = rel_pos
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # # qkv = self.to_qkv(x).split([self.dim, 2*self.dim], dim=-1)
        # # q, k, v = map(lambda t: rearrange(t, 'b n s (h d) -> b n s h d', h = self.heads), qkv)
        # # if s>1:
        # q, k, v = map(lambda t: rearrange(t, 'b n s (h d) -> b h n s d', h = self.num_heads), qkv)

        # lepe = self.lepe(rearrange(v, 'b h1 (j i) (h w) d -> b (h1 d) (j h) (i w)', j=self.num_wins, w=int(s**0.5)).contiguous())
        # lepe = self.lepe(rearrange(v, 'b h1 (j i) (h w) d -> b (j h) (i w) (h1 d)', j=self.num_wins, w=int(s**0.5)).contiguous())
        # km=rearrange(k, 'b h1 (j i) (h w) d -> b h1 (j h) (i w) d', j=self.num_wins, w=int(s**0.5))
        # vm=rearrange(v, 'b h1 (j i) (h w) d -> b h1 (j h) (i w) d', j=self.num_wins, w=int(s**0.5))
        km=rearrange(k, 'b h n s d -> b h (n s) d')
        vm=rearrange(v, 'b h n s d -> b h (n s) d')
        km=rearrange(km, 'b h (n p) d -> b h n p d', p=self.patchsize[1]**2)
        vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=self.patchsize[1]**2)


        # lepe = rearrange(lepe, 'n c h w -> n h w c')
        b,h,ns,s2,d=q.shape
        global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

        # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=self.patchsize[1],p2=self.patchsize[1]),dim=-2)
        global_tokenkm=torch.mean(km,dim=-2)
        global_tokenvm=torch.mean(vm,dim=-2)
        _,_,n1,_=global_tokenkm.shape
        _,_,n2,_=global_tokenk.shape
        dotsgs=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),km.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
        # dotsm=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenkm.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
        # dotsg=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenk.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
        sel0=(self.topk[0]*(self.patchsize[0]**2))
        sel1=(self.topk[1]*(self.patchsize[1]**2))
        sel2=(self.topk[2]*(self.patchsize[2]**2))

        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()
        if self.patchsize[0]==self.patchsize[1]:
            selectgs = all_indices[...,:sel0+sel1]
        else:
            selectgs = all_indices[...,:sel0]
            selectm = all_indices[...,sel0:sel0+sel1]
            selectm = extract_unique_elements(selectm//(self.patchsize[1]**2),self.topk[1])

        
        sel0=(self.topk[0])
        sel1=(self.topk[1])
        sel2=(self.topk[2])

        # selectg = torch.topk(dotsg,k=sel2,dim=-1)[-1]

        if self.patchsize[0]==self.patchsize[1]:
            offgs=torch.arange(0,b*n*self.num_heads*s,n*s,device='cuda')
            # offgs=torch.range(0,b*n*self.num_heads*s-1,n*s).long().cuda()
            # offm=torch.arange(0,b*n1*self.num_heads,n1).cuda()
            # off=torch.arange(0,b*n2*self.num_heads,n2,device='cuda')
            offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*(sel0+sel1))
            # offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*sel1)
            # off= repeat(off[:,None], 'd 1 -> (d b)', b = n*sel2)
        else:
            # print("pass")
            offgs=torch.arange(0,b*n*self.num_heads*s,n*s,device='cuda')
            # offgs=torch.arange(0,b*n*self.num_heads*s,n*s)
            offm=torch.arange(0,b*n1*self.num_heads,n1,device='cuda')
            # off=torch.arange(0,b*n2*self.num_heads,n2,device='cuda')
            offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*sel0)
            offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*sel1)
            # off= repeat(off[:,None], 'd 1 -> (d b)', b = n*sel2)

        if self.patchsize[0]==self.patchsize[1]:
            k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            # k2=torch.index_select(global_tokenk.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)
            # v2=torch.index_select(global_tokenv.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)
            k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=sel0+sel1,b=b,h=self.num_heads)
            v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=sel0+sel1,b=b,h=self.num_heads)
            # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
            # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
            if self.topk[2]:
                k = torch.cat((k0, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
                v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
            else:
                k=k0
                v=v0

        else:
            k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
            v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
            # k2=torch.index_select(global_tokenk.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)
            # v2=torch.index_select(global_tokenv.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)


            k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=sel0,b=b,h=self.num_heads)
            v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=sel0,b=b,h=self.num_heads)
            k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=sel1,b=b,h=self.num_heads)
            v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=sel1,b=b,h=self.num_heads)
            # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
            # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)

            # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
            v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
            # v = torch.cat((v0, v1, ), dim=-2)
            # v = v0

        dots=torch.matmul(q,k0.transpose(-1, -2))*self.scaling
        # print(q.shape)
        # print(local_k.shape)

        # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*self.scaling
        # print("dots",torch.isnan(dots).any().item())
        dotsm=torch.matmul(q,k1.transpose(-1, -2))*self.scaling
        # print("dots1",torch.isnan(dots1).any().item())
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*self.scaling
        # print("dotsg",torch.isnan(dotsg).any().item())


        # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(self.patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(self.patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(self.patchscale[2]))), dim=-1)
        if self.causal_mask:
            # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
            # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
            # indices=torch.range
            q_mask=torch.arange(0,n*s,1,device='cuda')
            q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
            # g_mask=torch.arange(0,n*s,s,device='cuda')
            g_mask=torch.arange(0,n,1,device='cuda')
            local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
            # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
            local_mask=torch.tril(local_mask, diagonal=s)

            # seq = torch.arange(n, device = 'cuda')
            b_t = rearrange(q_mask, 'n s -> 1 n s', n = n, s = s)
            bq_t = b_t
            bq_k = look_around1(b_t, **look_around_kwargs)

            bq_t = rearrange(bq_t, '... i -> ... i 1')
            bq_k = rearrange(bq_k, '... j -> ... 1 j')

            pad_mask = bq_k == -1
            causal_mask = bq_t < bq_k

            # g_mask = rearrange(g_mask, '(n s) -> n s',n=n,s=s)
            kv_mask=selectgs
            kv1_mask=selectm
            # kv1_mask=selectm*(self.patchsize[1]**2)
            mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
            maskm = (q_mask[None, None, :, :, None]//(self.patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
            maskg = (q_mask[:, :, None]//s) > g_mask[:]
            maskg=maskg[None, None, :, :]
            maskl=local_mask[None, None, None, :].bool()
            # maskl[...,0,:,:s]=False
            # print("q_mask",q_mask[:3,:])
            # print("kv_mask",kv_mask[0,0,:3,:])
            # print("mask",mask[0,0,:3,:3,:])
            # mask = F.pad(mask, (self.num_mem_kv, 0), value=True)
            # mask=rearrange(mask[...,sel0:sel0+sel1,:], 'b h n (t p) d -> b h n t p d',t=self.topk[1],)
            mask_value=-torch.finfo(k.dtype).max
            dots.masked_fill_(~mask, mask_value)
            # dotsl.masked_fill_(~maskl, mask_value)
            # dotsl.masked_fill_(causal_mask, mask_value)
            # dotsl = dotsl.masked_fill(pad_mask, mask_value)
            # print("mask",dots[0,0,:3,:3,:])
            dotsm.masked_fill_(~maskm, mask_value)
            dotsg.masked_fill_(~maskg, mask_value)
            del mask
            del maskm
            del maskg
            del maskl
            # g_mask=torch.arange(0,n*s,s,device='cuda')
        # dots=torch.cat((dots+torch.log((self.patchscale[0])), dots1+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
            
        # dots=torch.cat((dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
        dots=torch.cat((dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
        # dots=torch.cat((dotsl,dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)),), dim=-1)
        # dots=torch.cat((dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1))), dim=-1)
        # dots=torch.cat((dotsl,dots,), dim=-1)
        # print(self.patchscale[1],self.patchscale[2])

        # dots=torch.cat((dots+torch.log(torch.tensor(1)), dots1+torch.log(torch.tensor(1)), dotsg+torch.log(torch.tensor(1))), dim=-1)
        # attnscale=torch.ones((1,1,1,))
        # dots=torch.matmul(q,k.transpose(-1, -2))*self.scaling
        # sel0=(self.topk[0]*(self.patchsize[0]**2))
        # sel1=(self.topk[1]*(self.patchsize[1]**2))
        # sel2=(self.topk[2]*(self.patchsize[2]**2))
        # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(self.patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(self.patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(self.patchscale[2]))), dim=-1)

        # attn1=torch.exp(dots-torch.max(dots))
        # attn1=torch.cat((attn1[...,:sel0], attn1[...,sel0:sel0+sel1]*(self.patchsize[1]**2), attn1[...,sel0+sel1:]*s/2), dim=-1)
        # attn1=attn1/torch.sum(attn1,dim=-1)[...,None]
        # out=torch.matmul(self.dropout(attn1),v)
        attn=self.dropout(self.attend(dots))

        # out=torch.matmul(self.dropout(self.attend(dots)),v)
        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)
        # v=torch.cat((local_v,v),dim=-2)
        # v=local_v
        # out=torch.matmul(attn,v)

        # out = rearrange(out, 'b h (p1 p2) (h1 w1) d -> b (p1 h1) (p2 w1) (h d)',h=self.num_heads,p1=self.num_wins,p2=self.num_wins,w1=int(s**0.5))
        # out = (self.out_proj(out+lepe))
        # out = rearrange(out, 'b h n s d -> b (n s) (h d)')
        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))
        # out = rearrange(out, 'b h w d -> b d h w')

        return (out)

        # q = self.q_proj(x)
        # k = self.k_proj(x)
        # v = self.v_proj(x)
        # lepe = self.lepe(v)

        output = self.out_proj(output)
        return output


# import sys
# sys.path.insert(0, '/home/wangning/LongLoRA-main/llama_attn_replace.py')

class LlamaAttention_mss(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None):
        super().__init__()
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()


        self.scaling=self.head_dim**(-0.5)
        self.max_seq = 16*1024
        self.local_windows = 128
        self.topk = [128,64,64]
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,4,8,16]
        patchsize=[1,2,8,16]
        # self.patchscale = patchscale
        # self.patchscale1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        # self.attend=nn.Softmax(dim=-1)
        # self.dropout=nn.Dropout(0.)
        self.causal_mask=True
        # if patchscale[0]==0:
        
        patchscale=[]
        # patchscale.append(1.0*32*torch.ones((self.local_windows*2)))
        for i in range(len(self.topk)):
            patchscale.append(1.0*self.patchsize[i]*torch.ones((self.topk[i])))
        # patchscale.append(1.0*64*torch.ones((self.max_seq//self.local_windows)))
        patchscale=nn.Parameter(torch.cat(patchscale),requires_grad=True)
        self.patchscale=nn.ParameterList()
        self.patchscale.append(patchscale)
        self.patchscale.append(nn.Parameter(1.0*64*torch.ones((self.max_seq//self.local_windows)),requires_grad=True))
        # # self.patchscale=[]
        # # self.patchscale.append(nn.Parameter(torch.tensor(1.0),requires_grad=True))
        # # self.patchscale.append(nn.Parameter(torch.tensor(4.0,requires_grad=True),requires_grad=True))
        # # self.patchscale.append(nn.Parameter(torch.tensor(64.0,requires_grad=True),requires_grad=True))
        # self.patchscale.append(nn.Parameter(1.0*torch.ones((self.topk[0])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(2.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(8.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward_mss(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # print("LlamaAttention_mss")
        autopad=True

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if autopad:
            orig_seq_len = hidden_states.shape[-2]
            # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.local_windows, dim = -2), (q, k, v))
            (needed_pad, query_states), (_, key_states), (_, value_states) = map(lambda t: pad_to_multiple(t, self.local_windows, dim = -2), (query_states, key_states, value_states))
            # (_, hidden_states) = map(lambda t: pad_to_multiple(t, self.local_windows, dim = -2), hidden_states)

        q=rearrange(query_states, 'b h (n s) d -> b h n s d',s=self.local_windows, h = self.num_heads)
        k=rearrange(key_states, 'b h (n s) d -> b h n s d',s=self.local_windows, h = self.num_heads)
        v=rearrange(value_states, 'b h (n s) d -> b h n s d',s=self.local_windows, h = self.num_heads)
        b,h,n,s,d=q.shape

        km=rearrange(k, 'b h n s d -> b h (n s) d')
        vm=rearrange(v, 'b h n s d -> b h (n s) d')
        km=rearrange(km, 'b h (n p) d -> b h n p d', p=self.patchsize[1]**2)
        vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=self.patchsize[1]**2)

        global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

        # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=self.patchsize[1],p2=self.patchsize[1]),dim=-2)
        global_tokenkm=torch.mean(km,dim=-2)
        global_tokenvm=torch.mean(vm,dim=-2)
        _,_,n1,_=global_tokenkm.shape
        _,_,n2,_=global_tokenk.shape
        dotsgs=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),km.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
        # dotsm=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenkm.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
        # dotsg=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenk.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
        sel0=(self.topk[0]*(self.patchsize[0]**2))
        sel1=(self.topk[1]*(self.patchsize[1]**2))
        sel2=(self.topk[2]*(self.patchsize[2]**2))

        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()
        if self.patchsize[0]==self.patchsize[1]:
            selectgs = all_indices[...,:sel0+sel1]
        else:
            selectgs = all_indices[...,:sel0]
            selectm = all_indices[...,sel0:sel0+sel1]
            selectm = extract_unique_elements(selectm//(self.patchsize[1]**2),self.topk[1])

        
        sel0=(self.topk[0])
        sel1=(self.topk[1])
        sel2=(self.topk[2])

        # selectg = torch.topk(dotsg,k=sel2,dim=-1)[-1]

        if self.patchsize[0]==self.patchsize[1]:
            offgs=torch.arange(0,b*n*self.num_heads*s,n*s,device='cuda')
            # offgs=torch.range(0,b*n*self.num_heads*s-1,n*s).long().cuda()
            # offm=torch.arange(0,b*n1*self.num_heads,n1).cuda()
            # off=torch.arange(0,b*n2*self.num_heads,n2,device='cuda')
            offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*(sel0+sel1))
            # offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*sel1)
            # off= repeat(off[:,None], 'd 1 -> (d b)', b = n*sel2)
        else:
            # print("pass")
            offgs=torch.arange(0,b*n*self.num_heads*s,n*s,device='cuda')
            # offgs=torch.arange(0,b*n*self.num_heads*s,n*s)
            offm=torch.arange(0,b*n1*self.num_heads,n1,device='cuda')
            # off=torch.arange(0,b*n2*self.num_heads,n2,device='cuda')
            offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*sel0)
            offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*sel1)
            # off= repeat(off[:,None], 'd 1 -> (d b)', b = n*sel2)

        if self.patchsize[0]==self.patchsize[1]:
            k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            # k2=torch.index_select(global_tokenk.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)
            # v2=torch.index_select(global_tokenv.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)
            k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=sel0+sel1,b=b,h=self.num_heads)
            v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=sel0+sel1,b=b,h=self.num_heads)
            # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
            # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
            if self.topk[2]:
                k = torch.cat((k0, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
                v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
            else:
                k=k0
                v=v0

        else:
            k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
            k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
            v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
            # k2=torch.index_select(global_tokenk.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)
            # v2=torch.index_select(global_tokenv.reshape(-1,d),dim=0,index=selectg.reshape(-1)+off)


            k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=sel0,b=b,h=self.num_heads)
            v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=sel0,b=b,h=self.num_heads)
            k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=sel1,b=b,h=self.num_heads)
            v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=sel1,b=b,h=self.num_heads)
            # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
            # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)

            # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
            v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
            # v = torch.cat((v0, v1, ), dim=-2)
            # v = v0

        dots=torch.matmul(q,k0.transpose(-1, -2))*self.scaling
        # print(q.shape)
        # print(local_k.shape)

        # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*self.scaling
        # print("dots",torch.isnan(dots).any().item())
        dotsm=torch.matmul(q,k1.transpose(-1, -2))*self.scaling
        # print("dots1",torch.isnan(dots1).any().item())
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*self.scaling
        # print("dotsg",torch.isnan(dotsg).any().item())


        # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(self.patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(self.patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(self.patchscale[2]))), dim=-1)
        if self.causal_mask:
            # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
            # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
            # indices=torch.range
            q_mask=torch.arange(0,n*s,1,device='cuda')
            q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
            # g_mask=torch.arange(0,n*s,s,device='cuda')
            g_mask=torch.arange(0,n,1,device='cuda')
            local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
            # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
            local_mask=torch.tril(local_mask, diagonal=s)

            # seq = torch.arange(n, device = 'cuda')
            b_t = rearrange(q_mask, 'n s -> 1 n s', n = n, s = s)
            look_around_kwargs = dict(
                backward =  1,
                forward =  0,
                pad_value = -1
            )
            bq_t = b_t
            bq_k = look_around1(b_t, **look_around_kwargs)

            bq_t = rearrange(bq_t, '... i -> ... i 1')
            bq_k = rearrange(bq_k, '... j -> ... 1 j')

            pad_mask = bq_k == -1
            causal_mask = bq_t < bq_k

            # g_mask = rearrange(g_mask, '(n s) -> n s',n=n,s=s)
            kv_mask=selectgs
            kv1_mask=selectm
            # kv1_mask=selectm*(self.patchsize[1]**2)
            mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
            maskm = (q_mask[None, None, :, :, None]//(self.patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
            maskg = (q_mask[:, :, None]//s) > g_mask[:]
            maskg=maskg[None, None, :, :]
            maskl=local_mask[None, None, None, :].bool()
            # maskl[...,0,:,:s]=False
            # print("q_mask",q_mask[:3,:])
            # print("kv_mask",kv_mask[0,0,:3,:])
            # print("mask",mask[0,0,:3,:3,:])
            # mask = F.pad(mask, (self.num_mem_kv, 0), value=True)
            # mask=rearrange(mask[...,sel0:sel0+sel1,:], 'b h n (t p) d -> b h n t p d',t=self.topk[1],)
            mask_value=-torch.finfo(k.dtype).max
            dots.masked_fill_(~mask, mask_value)
            # dotsl.masked_fill_(~maskl, mask_value)
            # dotsl.masked_fill_(causal_mask, mask_value)
            # dotsl = dotsl.masked_fill(pad_mask, mask_value)
            # print("mask",dots[0,0,:3,:3,:])
            dotsm.masked_fill_(~maskm, mask_value)
            dotsg.masked_fill_(~maskg, mask_value)
            del mask
            del maskm
            del maskg
            del maskl
            # g_mask=torch.arange(0,n*s,s,device='cuda')
        # dots=torch.cat((dots+torch.log((self.patchscale[0])), dots1+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
            
        # dots=torch.cat((dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
        dots=torch.cat((dots+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.softmax(dots, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        out = rearrange(attn_output, 'b h n s d -> b h (n s) d')
        if autopad:
            attn_output = out[..., :orig_seq_len, :]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward4(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        group_size = int(q_len * group_size_ratio)

        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_group = q_len // group_size

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # shift
        # def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        #     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        #     qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        #     return qkv

        # query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        # key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        # value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        # query_r=query_states[:,self.num_heads//2:]
        # key_r=key_states[:,self.num_heads//2:]
        # value_r=value_states[:,self.num_heads//2:]

        # # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # query_l=query_states[:,:self.num_heads//2]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//4, self.head_dim)

        attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # print(attn_output.shape,attn_output2.shape)

        attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        group_size = int(q_len * group_size_ratio)

        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_group = q_len // group_size

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # shift
        # def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        #     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        #     qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        #     return qkv

        # query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        # key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        # value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        # query_r=query_states[:,self.num_heads//2:]
        # key_r=key_states[:,self.num_heads//2:]
        # value_r=value_states[:,self.num_heads//2:]

        # # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # query_l=query_states[:,:self.num_heads//2]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)

        #select1
        attn_output,attn_weights=forward_mss_local_inline_for(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def forward_o(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        group_size = int(q_len * group_size_ratio)

        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_group = q_len // group_size

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # shift
        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
            return qkv

        query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
            raise ValueError(
                f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
                f" {attn_weights.size()}"
            )

        # attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
        # if attention_mask is not None:
            if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
                raise ValueError(
                    f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        local_mask=torch.ones((group_size,group_size),dtype=torch.int).cuda()
        local_mask=torch.tril(local_mask, diagonal=0)
        local_mask = local_mask[None, None, :group_size, :group_size].repeat(bsz * num_group, self.num_heads, 1, 1).bool()
        mask_value=-torch.finfo(key_states.dtype).max
        attn_weights.masked_fill_(~local_mask, mask_value)
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

        # shift back
        attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




def forward_mss(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    topklist=[256,64,64]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(2.0),torch.tensor(50.0)]
    local_windows=64
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if autopad:
        orig_seq_len = hidden_states.shape[-2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad, query_states), (_, key_states), (_, value_states) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (query_states, key_states, value_states))
        # (_, hidden_states) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), hidden_states)
    q,k,v=query_states,key_states,value_states
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn, past_key_value

    # local_windows=128
    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b (n s) (h d) -> b h n s d',n=local_windows, h = self.num_heads)
    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows, h = self.num_heads)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows, h = self.num_heads)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows, h = self.num_heads)
    b,h,n,s,d=q.shape
    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),km.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenkm.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenk.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*self.num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=self.num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=self.num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn, past_key_value



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*self.num_heads,n1,device='cuda')
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=self.num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=self.num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)


def forward_local_mss(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    topklist=[128,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(4.0),torch.tensor(10.0)]
    local_windows=128
    scaling=self.head_dim**0.5
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if autopad:
        orig_seq_len = hidden_states.shape[-2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad, query_states), (_, key_states), (_, value_states) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (query_states, key_states, value_states))
        # (_, hidden_states) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), hidden_states)
    q,k,v=query_states,key_states,value_states
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn, past_key_value

    local_windows=64
    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b (n s) (h d) -> b h n s d',n=local_windows, h = self.num_heads)
    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows, h = self.num_heads)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows, h = self.num_heads)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows, h = self.num_heads)
    b,h,n,s,d=q.shape
    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),km.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenkm.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,self.num_heads,-1,d),global_tokenk.reshape(b,self.num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*self.num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=self.num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=self.num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn, past_key_value



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*self.num_heads,n1,device='cuda')
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=self.num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=self.num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=self.num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)

def forward_mss_inline_own(self,q,k,v):
    # bsz, q_len, _ = q.size()
    topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)

    look_around_kwargs = dict(
        backward =  1,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*num_heads,n1,device='cuda')
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,2*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots[...,:sel0]+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dots[...,sel0:sel0+sel1]+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)+1), dots[...,sel0+sel1:]+torch.log((self.patchscale[2])%(s)+1)), dim=-1)
    dots=torch.cat((dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = rearrange(attn_output, 'b n h d -> b n (h d)')

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

    return attn_output, attn_weights

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)

def forward_mss_local_inline_own(self,q,k,v,topklist=[128,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    # print(local_k.shape)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    km=rearrange(k, 'b h n s d -> b h (n s) d')
    vm=rearrange(v, 'b h n s d -> b h (n s) d')
    km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    _,_,n1,_=global_tokenkm.shape
    _,_,n2,_=global_tokenk.shape
    dotsgqs=torch.matmul(global_tokenq[...,None,:],q.transpose(-1, -2))*scaling
    dotsgqs=nn.functional.softmax(dotsgqs, dim=-1)
    global_tokenq=dotsgqs@q

    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),km.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))

    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda')
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*num_heads,n1,device='cuda')
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((local_v,v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print(dotsl.shape)       
    # print(dots.shape)       
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,(backward+1)*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=backward*s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots[...,:sel0]+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dots[...,sel0:sel0+sel1]+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)+1), dots[...,sel0+sel1:]+torch.log((self.patchscale[2])%(s)+1)), dim=-1)
    # dots=torch.cat((dotsl, dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dotsl, dots, dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = rearrange(attn_output, 'b n h d -> b n (h d)')

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

    return attn_output, attn_weights

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)


def forward_mss_local_inline_for(self,q,k,v,topklist=[128,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    # print(local_k.shape)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    # km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    # vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    k=rearrange(k, 'b h n s d -> b h (n s) d')
    v=rearrange(v, 'b h n s d -> b h (n s) d')

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),k.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))
    sel0=(topklist[0]*(patchsize[0]**2))
    sel1=(topklist[1]*(patchsize[1]**2))
    sel2=(topklist[2]*(patchsize[2]**2))
    sel_num=torch.sum(torch.tensor(topklist)*torch.tensor(patchsize))
    all_indices = torch.topk(dotsgs,k=int(sel_num),dim=-1)[-1].cuda()
    Kt=[]
    Vt=[]
    Mask=[]

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1).cuda()
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    k0,v0,mask0=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[0],patchsize=patchsize[0],selectlist=all_indices[...,:topklist[0]*patchsize[0]])
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)

    for i in range(1,len(topklist)):
        ki,vi,maski=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[i],patchsize=patchsize[i],selectlist=all_indices[...,topklist[i-1]*patchsize[i-1]:topklist[i]*patchsize[i]])
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1)
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out), attn


    if n*s<=sel0+sel1:
        all_indices = torch.topk(dotsgs,k=sel0,dim=-1)[-1].cuda()
    else:
        all_indices = torch.topk(dotsgs,k=sel0+sel1,dim=-1)[-1].cuda()


    
    # sel0=(topklist[0])
    # sel1=(topklist[1])
    # sel2=(topklist[2])

    selectgs = all_indices[...,:sel0]

    offgs=torch.arange(0,b*n*num_heads*s,n*s,device='cuda',dtype=k.dtype)
    offgs= repeat(offgs[:,None], 'd 1 -> (d b)', b = n*topklist[0])

    k0=torch.index_select(km.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    v0=torch.index_select(vm.reshape(-1,d),dim=0,index=selectgs.reshape(-1)+offgs)
    k0 = rearrange(k0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    v0 = rearrange(v0, '(b h n ki) d -> b h n (ki) d',ki=topklist[0],b=b,h=num_heads)
    if n*s<=sel0+sel1:
        v = torch.cat((v0, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
        dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
        dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        kv_mask=selectgs
        g_mask=torch.arange(0,n,1,device='cuda')

        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        # dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskg

        dots=torch.cat((dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)

        # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
        out=torch.matmul(attn,v)

        out = rearrange(out, 'b h n s d -> b h (n s) d')
        if autopad:
            out = out[..., :orig_seq_len, :]
        # out = (self.out_proj(out))

        return (out), attn



    selectm = all_indices[...,sel0:sel0+sel1]
    selectm = extract_unique_elements(selectm//(patchsize[1]**2),topklist[1])        
    offm=torch.arange(0,b*n1*num_heads,n1,device='cuda',dtype=k.dtype)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[1])
    k1=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v1=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)        
    k1 = rearrange(k1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    v1 = rearrange(v1, '(b h n ki) d -> b h n (ki) d',ki=topklist[1],b=b,h=num_heads)
    # k2 = rearrange(k2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)
    # v2 = rearrange(v2, '(b h n ki) d -> b h n (ki) d',ki=sel2,b=b,h=num_heads)

    # k = torch.cat((k0, k1, repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    v = torch.cat((local_v,v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print(dotsl.shape)       
    # print(dots.shape)       
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda',dtype=k.dtype)
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda',dtype=k.dtype)
        local_mask=torch.ones((s,(backward+1)*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=backward*s)

        kv_mask=selectgs
        kv1_mask=selectm
        # kv1_mask=selectm*(patchsize[1]**2)
        mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots[...,:sel0]+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dots[...,sel0:sel0+sel1]+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)+1), dots[...,sel0+sel1:]+torch.log((self.patchscale[2])%(s)+1)), dim=-1)
    # dots=torch.cat((dotsl, dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dotsl, dots, dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = rearrange(attn_output, 'b n h d -> b n (h d)')

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

    return attn_output, attn_weights

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)

def Select_KV1(self,shape,q_mask,k,v,topklist,patchsize,selectlist):
    b,h,n,s,d=shape
    num_heads=h

    km=rearrange(k, 'b h (n p) d -> b h n p d', p=patchsize)
    vm=rearrange(v, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkm=torch.mean(km,dim=-2)
    global_tokenvm=torch.mean(vm,dim=-2)
    nk=global_tokenkm.shape[-2]

    # selectm = all_indices[...,sel0:sel0+sel1]
    selectm = selectlist
    selectm = extract_unique_elements(selectm//(patchsize),topklist)

    offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    k=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    v=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)

    k = rearrange(k, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=num_heads)
    v = rearrange(v, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=num_heads)

    maskm = (q_mask[None, None, :, :, None]//(patchsize)) > selectm[:, :, :, None, :]

    return k,v,maskm


def Select_KV(self,shape,q_mask,k,v,topklist,patchsize,selectlist):
    b,h,n,s,d=shape
    num_heads=h

    if patchsize>1:
        km=rearrange(k, 'b h (n p) d -> b h n p d', p=patchsize)
        vm=rearrange(v, 'b h (n p) d -> b h n p d', p=patchsize)
        global_tokenkm=torch.mean(km,dim=-2)
        global_tokenvm=torch.mean(vm,dim=-2)
        nk=global_tokenkm.shape[-2]

        # selectm = all_indices[...,sel0:sel0+sel1]
        selectm = selectlist
        selectm = extract_unique_elements(selectm//(patchsize),topklist)
        maskm = (q_mask[None, None, :, :, None]//(patchsize)) > selectm[:, :, :, None, :]
    else:
        global_tokenkm=k
        global_tokenvm=v
        nk=global_tokenkm.shape[-2]
        selectm = selectlist
        maskm = (q_mask[None, None, :, :, None]//(patchsize)) >= selectm[:, :, :, None, :]

    offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    km=torch.index_select(global_tokenkm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)
    vm=torch.index_select(global_tokenvm.reshape(-1,d),dim=0,index=selectm.reshape(-1)+offm)

    km = rearrange(km, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=num_heads)
    vm = rearrange(vm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=num_heads)

    

    return km,vm,maskm

def forward_mss_local_inline_for2(self,q,k,v,topklist=[128,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    # print(local_k.shape)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    # km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    # vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    k=rearrange(k, 'b h n s d -> b h (n s) d')
    v=rearrange(v, 'b h n s d -> b h (n s) d')

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),k.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))


    sel_num=torch.sum(torch.tensor(topklist)*torch.tensor(patchsize))
    all_indices = torch.topk(dotsgs,k=int(sel_num),dim=-1)[-1].cuda()
    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s),dtype=torch.int).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1).cuda()
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    k0,v0,mask0=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[0],patchsize=patchsize[0],selectlist=all_indices[...,:topklist[0]*patchsize[0]])
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)

    for i in range(1,len(topklist)):
        ki,vi,maski=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[i],patchsize=patchsize[i],selectlist=all_indices[...,topklist[i-1]*patchsize[i-1]:topklist[i]*patchsize[i]])
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
    
    g_mask=torch.arange(0,n,1).cuda()
    maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1)
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out), attn




def forward_mss_local_inline_for1(self,q,k,v,topklist=[64,64,64,64,64],patchsize=[1,2,4,8,16]):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True
    # print(self.patchscale.data[:5])
    # print(self.patchscale[256:260])
    # print(self.patchscale.data[320:325])

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    # if (topklist[0]*(patchsize[0]**2))>=n:
    #     dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    #     q_mask=torch.arange(0,n,1,device='cuda')
    #     k_mask=torch.arange(0,n,1,device='cuda')
    #     mask = q_mask[:, None] >= k_mask[None, :]
    #     mask=mask[None, None, :, :]
    #     mask_value=-torch.finfo(k.dtype).max
    #     dots.masked_fill_(~mask, mask_value)
    #     # attn=self.dropout(self.attend(dots))
    #     attn=nn.functional.softmax(dots, dim=-1)
    #     out=torch.matmul(attn,v)
    #     return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    # print(local_k.shape)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    # km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    # vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    k=rearrange(k, 'b h n s d -> b h (n s) d')
    v=rearrange(v, 'b h n s d -> b h (n s) d')

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),k.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))


    mask_value=-torch.finfo(k.dtype).max
    sel_num=torch.sum(torch.tensor(topklist)*torch.tensor(patchsize))
    # print("sel_num",sel_num)
    # print("k",dotsgs.shape[-1])
    all_indices = torch.topk(dotsgs,k=int(sel_num),dim=-1)[-1].cuda()
    Kt=[]
    Vt=[]
    Mask=[]

    dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    local_mask=torch.tril(local_mask, diagonal=backward*s)
    maskl=local_mask[None, None, None, :].bool()
    dotsl.masked_fill_(~maskl, mask_value)
    del maskl


    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1).cuda()
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    k0,v0,mask0=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[0],patchsize=patchsize[0],selectlist=all_indices[...,:topklist[0]*patchsize[0]])
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[i],patchsize=patchsize[i],selectlist=all_indices[...,indx:indx+topklist[i]*patchsize[i]])
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    g_mask=torch.arange(0,n,1).cuda()
    maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    dotsg.masked_fill_(~maskg, mask_value)
    del maskg
    # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    del kv_mask
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    if self.debug<2:
        print(self.patchscale[0])
        print(self.patchscale[-1])
        self.debug+=1
    v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    out = out.transpose(1, 2).contiguous()

    return (out), attn


def forward_mss_local_inline_for3(self,q,k,v,topklist=[128,64,64],patchsize=[1,4,16]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True
    # print(self.patchscale.data[:5])
    # print(self.patchscale[256:260])
    # print(self.patchscale.data[320:325])

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    # if (topklist[0]*(patchsize[0]**2))>=n:
    #     dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    #     q_mask=torch.arange(0,n,1,device='cuda')
    #     k_mask=torch.arange(0,n,1,device='cuda')
    #     mask = q_mask[:, None] >= k_mask[None, :]
    #     mask=mask[None, None, :, :]
    #     mask_value=-torch.finfo(k.dtype).max
    #     dots.masked_fill_(~mask, mask_value)
    #     # attn=self.dropout(self.attend(dots))
    #     attn=nn.functional.softmax(dots, dim=-1)
    #     out=torch.matmul(attn,v)
    #     return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    # print(local_k.shape)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    # km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    # vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    k=rearrange(k, 'b h n s d -> b h (n s) d')
    v=rearrange(v, 'b h n s d -> b h (n s) d')

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),k.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))


    mask_value=-torch.finfo(k.dtype).max
    sel_num=torch.sum(torch.tensor(topklist)*torch.tensor(patchsize))
    all_indices = torch.topk(dotsgs,k=int(sel_num),dim=-1)[-1].cuda()
    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl


    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    Kt.append(local_k)
    Vt.append(local_v)
    Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1).cuda()
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    k0,v0,mask0=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[0],patchsize=patchsize[0],selectlist=all_indices[...,:topklist[0]*patchsize[0]])
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)

    for i in range(1,len(topklist)):
        ki,vi,maski=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[i],patchsize=patchsize[i],selectlist=all_indices[...,topklist[i-1]*patchsize[i-1]:topklist[i]*patchsize[i]])
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
    
    g_mask=torch.arange(0,n,1).cuda()
    maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    del kv_mask
    attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots,dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    out = out.transpose(1, 2).contiguous()

    return (out), attn



def forward_mss_local_inline_ownf(self,q,k,v,topklist=[128,128,128]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    local_windows=128
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    
    # q = self.q_proj(x)
    # k = self.k_proj(x)
    # v = self.v_proj(x)
    n=q.shape[-2]
    if (topklist[0]*(patchsize[0]**2))>=n:
        dots=torch.matmul(q,k.transpose(-1, -2))*scaling
        q_mask=torch.arange(0,n,1,device='cuda')
        k_mask=torch.arange(0,n,1,device='cuda')
        mask = q_mask[:, None] >= k_mask[None, :]
        mask=mask[None, None, :, :]
        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        # attn=self.dropout(self.attend(dots))
        attn=nn.functional.softmax(dots, dim=-1)
        out=torch.matmul(attn,v)
        return out, attn

    if autopad:
        orig_seq_len = q.shape[2]
        (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    b,h,n,s,d=q.shape
    num_heads=h

    dots_shortcut=torch.sum(q*k*scaling,dim=-1)
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    # print(local_k.shape)       
    # local_k= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),k[...,:-1,:,:]),dim=-3), k), dim=-2)
    # local_v= torch.cat((torch.cat((-torch.ones((b,h,1,s,d)).cuda(),v[...,:-1,:,:]),dim=-3), v), dim=-2)

    # pos_emb, xpos_scale = self.rel_pos(local_k)
    # q, local_k = apply_rotary_pos_emb(q, local_k, pos_emb, scale = xpos_scale)
    
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    # km=rearrange(km, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)
    # vm=rearrange(vm, 'b h (n p) d -> b h n p d', p=patchsize[1]**2)


    # lepe = rearrange(lepe, 'n c h w -> n h w c')
    b,h,ns,s2,d=q.shape
    global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    k=rearrange(k, 'b h n s d -> b h (n s) d')
    v=rearrange(v, 'b h n s d -> b h (n s) d')

    # global_tokenkm=torch.mean(rearrange(km, 'b h (h1 p1) (w1 p2) d -> b h (h1 w1) (p1 p2) d', p1=patchsize[1],p2=patchsize[1]),dim=-2)
    dotsgs=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),k.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsm=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenkm.reshape(b,num_heads,-1,d).transpose(-1, -2))
    # dotsg=torch.matmul(global_tokenq.reshape(b,num_heads,-1,d),global_tokenk.reshape(b,num_heads,-1,d).transpose(-1, -2))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    sel_num=torch.sum(torch.tensor(topklist)*torch.tensor(patchsize))
    all_indices = torch.topk(dotsgs,k=int(sel_num),dim=-1)[-1].cuda()

    k0,v0,mask0=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[0],patchsize=patchsize[0],selectlist=all_indices[...,:topklist[0]*patchsize[0]])

    k1,v1,mask1=Select_KV(self,q.shape,q_mask,k,v,topklist=topklist[1],patchsize=patchsize[1],selectlist=all_indices[...,topklist[0]*patchsize[0]:topklist[1]*patchsize[1]])
    
    v = torch.cat((local_v,v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0

    dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # print(q.shape)
    # print(local_k.shape)

    dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # print(dotsl.shape)       
    # print(dots.shape)       
    # print("dots",torch.isnan(dots).any().item())
    dotsm=torch.matmul(q,k1.transpose(-1, -2))*scaling
    # print("dots1",torch.isnan(dots1).any().item())
    dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # print("dotsg",torch.isnan(dotsg).any().item())


    # dots=torch.cat((dots[...,:sel0]+torch.log(torch.tensor(patchscale[0])), dots[...,sel0:sel0+sel1]+torch.log(torch.tensor(patchscale[1])), dots[...,sel0+sel1:]+torch.log(torch.tensor(patchscale[2]))), dim=-1)
    if causal_mask:
        # q_mask, kv_mask = map(lambda t: t.reshape(b, h, ns, -1), (indices, all_indices))
        # mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
        # indices=torch.range
        q_mask=torch.arange(0,n*s,1,device='cuda')
        q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
        g_mask=torch.arange(0,n,1,device='cuda')
        local_mask=torch.ones((s,(backward+1)*s),dtype=torch.int).cuda()
        # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
        local_mask=torch.tril(local_mask, diagonal=backward*s)

        # kv_mask=selectgs
        # kv1_mask=selectm
        # # kv1_mask=selectm*(patchsize[1]**2)
        # mask = q_mask[None, None, :, :, None] >= kv_mask[:, :, :, None, :]
        # maskm = (q_mask[None, None, :, :, None]//(patchsize[1]**2)) > kv1_mask[:, :, :, None, :]
        mask = mask0
        maskm = mask1
        maskg = (q_mask[:, :, None]//s) > g_mask[:]
        maskg=maskg[None, None, :, :]
        maskl=local_mask[None, None, None, :].bool()

        mask_value=-torch.finfo(k.dtype).max
        dots.masked_fill_(~mask, mask_value)
        dotsm.masked_fill_(~maskm, mask_value)
        dotsg.masked_fill_(~maskg, mask_value)
        dotsl.masked_fill_(~maskl, mask_value)
        # dotsl.masked_fill_(causal_mask, mask_value)
        # dotsl = dotsl.masked_fill(pad_mask, mask_value)
        del mask
        del maskm
        del maskg
        del maskl
    # dots=torch.cat((dots+torch.log((patchscale[0])), dots1+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)            
    # dots=torch.cat((dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots_shortcut[...,None],dots+torch.log((patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((patchscale[2])%(s+1))), dim=-1)
    # dots=torch.cat((dots[...,:sel0]+torch.log((self.patchscale[0])%(self.patchsize[0]**2+1)), dots[...,sel0:sel0+sel1]+torch.log((self.patchscale[1])%(self.patchsize[1]**2+1)+1), dots[...,sel0+sel1:]+torch.log((self.patchscale[2])%(s)+1)), dim=-1)
    # dots=torch.cat((dotsl, dots+torch.log((self.patchscale[0])%(patchsize[0]**2+1)), dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
    dots=torch.cat((dotsl, dots, dotsm+torch.log((self.patchscale[1])%(patchsize[1]**2+1)), dotsg+torch.log((self.patchscale[2])%(s+1))), dim=-1)
    # print(self.patchscale[1][:5],self.patchscale[2][:5])
    # print(self.patchscale[1].requires_grad)
    # print(self.patchscale[2])

    # attn=self.dropout(self.attend(dots))
    attn_weights = nn.functional.softmax(dots, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output = torch.matmul(attn_weights, value_states)

    attn_output = rearrange(attn_output, 'b h n s d -> b h (n s) d')
    if autopad:
        attn_output = attn_output[..., :orig_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = rearrange(attn_output, 'b n h d -> b n (h d)')

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

    return attn_output, attn_weights

    # out=torch.matmul(attn[...,1:],v)+attn[...,0][...,None]*v_shortcut
    out=torch.matmul(attn,v)

    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))

    return (out)

