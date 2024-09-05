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
# from flash_attn.flash_blocksparse_attn_interface import flash_blocksparse_attn_func
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_func
)

from transformers.models.llama.modeling_llama import repeat_kv,rotate_half,apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding, LlamaAttention
from flash_attn.bert_padding import unpad_input, pad_input
import math
from transformers.models.llama.configuration_llama import LlamaConfig
# from transformers.models.mistral.configuration_mistral import MistralConfig
# from local_attention import LocalAttention
group_size_ratio=1/16
# from ring_attention_pytorch import ring_flash_attn

# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
#     # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
#     cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
#     sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
#     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def forward_LBe(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    # assert past_key_value is None, "past_key_value is not supported"
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    # assert not use_cache, "use_cache is not supported"
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    past_key_value = (key_states, value_states) if use_cache else None

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    # import pdb; pdb.set_trace()

    if query_states.shape[-2] == 1 or query_states.shape[-2] != key_states.shape[-2]:
        # decode token-by-token, do not use flash attention
        # in incremental state, do not use flash attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # print("query_states",query_states.shape)
    # print("key_states",key_states.shape)
    # print("value_states",value_states.shape)
    qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                 device=qkv.device)
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                     indices, bsz, q_len),
                           'b s (h d) -> b s h d', h=nheads)
    attn_output = self.o_proj(rearrange(output, 'b s h d -> b s (h d)'))

    return attn_output, None, past_key_value




def process_attention_layer_in_chunks(attention_layer, queries, keys, values, chunk_size=int(64*1024)):
    num_heads = queries.size(2)  # Assuming the queries have shape [batch_size, seq_length, num_heads, head_dim]
    head_dim = queries.size(-1)
    
    # attn_output_his = torch.zeros_like(queries).cpu()  # Initialize history to accumulate attention outputs
    # Slsek = torch.full((queries.size(0), num_heads), float('-inf')).cpu()  # Initialize Slsek to negative infinity
    attn_output_his = None  # Initialize history to accumulate attention outputs
    Slsek = None  # Initialize Slsek to negative infinity
    chunk_outputs = []
    
    for i in range(0, queries.size(1), chunk_size):
        q_chunk = queries[:, i:i+chunk_size].to('cuda')
        # chunk_outputs = []
        attn_output_his = None  # Initialize history to accumulate attention outputs
        Slsek = None  # Initialize Slsek to negative infinity
        # print("i",i)
        
        for j in range(0, i+chunk_size-1, chunk_size):
            k_chunk = keys[:, j:j+chunk_size].to('cuda')
            v_chunk = values[:, j:j+chunk_size].to('cuda')
            # print("k_chunk",k_chunk.shape)
            # print("i",i)
            # print("j",j)
            
            # Perform attention computation
            # attn_output, attn_scores = attention_layer(q_chunk, k_chunk, v_chunk, need_weights=True)
            if j==i:
                attn_output, slek, _ = flash_attn_func(q_chunk, k_chunk, v_chunk, return_attn_probs=True, causal=True)
            else:
                attn_output, slek, _ = flash_attn_func(q_chunk, k_chunk, v_chunk, return_attn_probs=True)
            
            # Move the result to CPU to accumulate
            # attn_output = attn_output
            # slek = attn_scores.logsumexp(dim=-1)
            attn_output=attn_output.transpose(1,2)
            
            # LogSumExp trick to update Slsek and attn_output_his
            # print("Slsek",Slsek)
            if Slsek is None:
                Slsek = slek
                attn_output_his = attn_output
            else:
                lsesk = torch.logsumexp(torch.cat((slek[..., None], Slsek[..., None]), dim=-1), dim=-1).detach()
                attn_output_his = attn_output_his * (1 / (torch.exp(lsesk - Slsek)))[..., None] + attn_output * (1 / (torch.exp(lsesk - slek)))[..., None]
                Slsek = lsesk
        
        chunk_outputs.append(attn_output_his.transpose(1,2).cpu())
        # print("attn_output_his",attn_output_his.shape)
    
    return torch.cat(chunk_outputs, dim=1)
def process_linear_layer_in_chunks(layer, inputs, chunk_size=int(128*1024)):
    outputs = []
    for i in range(0, inputs.size(1), chunk_size):
        chunk = inputs[:, i:i+chunk_size]
        chunk_output = layer(chunk.to('cuda'))
        outputs.append(chunk_output.cpu())
        # del chunk_output
    return torch.cat(outputs, dim=1)

def forward_LBc(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    # query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = process_linear_layer_in_chunks(self.q_proj,hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = process_linear_layer_in_chunks(self.k_proj,hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = process_linear_layer_in_chunks(self.v_proj,hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    # assert past_key_value is None, "past_key_value is not supported"
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, position_ids)
    # print(position_ids.shape)
    # print(cos.shape)
    # print(query_states.shape)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    # assert not use_cache, "use_cache is not supported"
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    past_key_value = (key_states, value_states) if use_cache else None

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    # import pdb; pdb.set_trace()


    attn_output=process_attention_layer_in_chunks(None,query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2))
    # print(attn_output.shape)
    # attn_output=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True)
    attn_weights=None
    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[4],patchsize=[2048],w=2048,flash_use=True)
    # print(self.layer_idx)
    # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

    # shift back
    # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # attn_output = self.o_proj(attn_output)
    # print(attn_output.dtype)
    attn_output = process_linear_layer_in_chunks(self.o_proj,attn_output.to(torch.bfloat16))
    # print(attn_output.dtype)


    if not output_attentions:
        attn_weights = None

    return attn_output, None, past_key_value




def forward_ring_flashattne(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)


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
    # assert past_key_value is None, "past_key_value is not supported"
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    # assert not use_cache, "use_cache is not supported"
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

    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[1024],patchsize=[32],w=32,mergep=512,flash_use=True)
    # attn_output=ring_flash_attn(query_states.to(torch.bfloat16).transpose(1, 2),key_states.to(torch.bfloat16).transpose(1, 2),value_states.to(torch.bfloat16).transpose(1, 2),causal=True,bucket_size=1024)
    attn_output=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True)
    attn_weights=None
    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[4],patchsize=[2048],w=2048,flash_use=True)
    # print(self.layer_idx)
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


def forward_flashattn(
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
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim)
        return qkv

    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

    attn_output=flash_attn_func(query_states,key_states,value_states,causal=True)


    if attn_output.size() != (bsz * num_group, group_size, self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

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


def forward_flashattn1(
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

    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt_seg(self,query_states,key_states,value_states,topklist=[4],patchsize=[2048],w=2048,flash_use=True,seg=10240)
    attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[512],patchsize=[16],w=16,mergep=128,flash_use=True)
    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[4],patchsize=[2048],w=2048,flash_use=True)
    # print(self.layer_idx)
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


def forward_flashattne(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)


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
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[1024],patchsize=[32],w=32,mergep=512,flash_use=True)
    attn_output=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True)
    attn_weights=None
    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[4],patchsize=[2048],w=2048,flash_use=True)
    # print(self.layer_idx)
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

def forward_flashattne2(
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
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[1024],patchsize=[32],w=32,mergep=512,flash_use=True)
    chunksize=4096
    attn_outputs=[]
    for i in range((kv_seq_len-1)//chunksize):
        attn_output=flash_attn_func(query_states[...,i*chunksize:(i+1)*chunksize,:].transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True)
        attn_outputs.append(attn_output)
    attn_output=flash_attn_func(query_states[...,((kv_seq_len-1)//chunksize)*chunksize:,:].transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True)
    attn_outputs.append(attn_output)
    attn_output=torch.cat(attn_outputs,dim=1)

    attn_weights=None
    # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[4],patchsize=[2048],w=2048,flash_use=True)
    # print(self.layer_idx)
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



def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def pad_to_multiple_t(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, remainder, 0), value = value)

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
    # padded_x = F.pad(x, (*dims, backward, forward), mode='replicate')
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



def Select_KV_C_mixc1(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16):
    # b,h,n,s,d=shape
    # num_heads=h
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]

    
    # print(km.shape)
    # print(global_tokenkm.shape)
    b,h,n,s,w,d=shape
    if nkv>seg_len:
        (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, h*n, dim = -2)
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device='cuda',dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b = b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = rearrange(global_tokenkvm, 'b h (n p) d -> b h n p d', p=seg_len)
        p=global_tokenkvm.shape[-2]
        global_tokenkvm = look_around(global_tokenkvm, **look_around_kwargs)
        
    else:        
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device='cuda',dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = global_tokenkvm[:,:,None]
        nkv=global_tokenkvm.shape[-2]

    if n>=patchsize:
        global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
    else:
        global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d', p=patchsize)
        global_tokenq = global_tokenq[:,:,None]

    dots_s=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots_s,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*n1*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = s1*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n1 s ki) d -> b h n1 s (ki) d',ki=topklist,b=b,h=h,s=s)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    if patchsize==1:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    else:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) > maskm[..., None, :]
    # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16):
    # b,h,n,s,d=shape
    # num_heads=h
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]

    
    # print(km.shape)
    # print(global_tokenkm.shape)
    b,h,n,s,w,d=shape
    if nkv>seg_len:
        (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device='cuda',dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b = b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = rearrange(global_tokenkvm, 'b h (n p) d -> b h n p d', p=seg_len)
        p=global_tokenkvm.shape[-2]
        global_tokenkvm = look_around(global_tokenkvm, **look_around_kwargs)
        
    else:        
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device='cuda',dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = global_tokenkvm[:,:,None]
        nkv=global_tokenkvm.shape[-2]

    if n>=patchsize:
        global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
    else:
        global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
        global_tokenq = global_tokenq[:,:,None]

    dots_s=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots_s,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*n1*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = s1*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n1 s ki) d -> b h n1 s (ki) d',ki=topklist,b=b,h=h,s=s)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    if patchsize==1:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    else:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) > maskm[..., None, :]
    # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc_cross(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16):
    # b,h,n,s,d=shape
    # num_heads=h
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]

    
    # print(km.shape)
    # print(global_tokenkm.shape)
    b,h,n,s,w,d=shape
    if nkv>seg_len:
        (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device='cuda',dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b = b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = rearrange(global_tokenkvm, 'b h (n p) d -> b h n p d', p=seg_len)
        p=global_tokenkvm.shape[-2]
        global_tokenkvm = look_around(global_tokenkvm, **look_around_kwargs)
        
    else:        
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device='cuda',dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = global_tokenkvm[:,:,None]
        nkv=global_tokenkvm.shape[-2]

    if n>=patchsize:
        global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
    else:
        global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
        global_tokenq = global_tokenq[:,:,None]

    dots_s=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots_s,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*n1*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = s1*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n1 s ki) d -> b h n1 s (ki) d',ki=topklist,b=b,h=h,s=s)
    kvm2=rearrange(kvm, 'b h n1 s ki d -> b h (n1 s ki) d')
    kvm2 = torch.cat((kvm2[...,:topklist,:],kvm2[...,:-topklist,:]),dim=-2)
    kvm2=rearrange(kvm2, 'b h (n1 s ki) d -> b h n1 s ki d',ki=topklist,b=b,h=h,s=s)
    kvm=torch.cat((kvm2,kvm),dim=-2)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    # print(kvm.shape)
    if patchsize==1:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    else:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) > maskm[..., None, :]
    # print(maskm.shape)
    # km2=rearrange(km, 'b h n1 s ki d -> b h (n1 s ki) d')
    # km2 = torch.cat((km2[...,:topklist,:],km[...,:-topklist,:]),dim=-2)
    # km2=rearrange(km2, 'b h (n1 s ki) d -> b h n1 s ki d',ki=topklist,b=b,h=h,s=s)
    # print(maskm.shape)
    # km=torch.cat((km2,km),dim=-2)

    

    return km,vm,maskm


def Select_KV_C_mixc_kvpre(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,kv_pre=None,indx=0):
    # b,h,n,s,d=shape
    # num_heads=h
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]

    
    # print(km.shape)
    # print(global_tokenkm.shape)
    b,h,n,s,w,d=shape
    # assert 
    # if nkv>seg_len:
    if n>=patchsize and nkv>seg_len:
        global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
        (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device=global_tokenkvm.device,dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b = b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)

        # (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        
        global_tokenkvm = rearrange(global_tokenkvm, 'b h (n p) d -> b h n p d', p=seg_len)
        p=global_tokenkvm.shape[-2]
        global_tokenkvm = look_around(global_tokenkvm, **look_around_kwargs)
        if kv_pre is not None:
            # global_tokenkvm[...,0,:seg_len,:d*2]=torch.mean(rearrange(kv_pre, 'b h (n p) d -> b h n p d', p=patchsize),dim=-2)
            global_tokenkvm[...,0,:seg_len,:d*2]=kv_pre
    else:
        global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
        global_tokenq = global_tokenq[:,:,None]
        if kv_pre is not None:
            global_tokenkvm=torch.cat((kv_pre,global_tokenkvm),dim=-2)        
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device=global_tokenkvm.device,dtype=torch.int)-seg_len
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = global_tokenkvm[:,:,None]
        nkv=global_tokenkvm.shape[-2]
        

    # if n>=patchsize:
    #     global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
    # else:
    #     global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
    #     global_tokenq = global_tokenq[:,:,None]

    # print(global_tokenq.shape)
    # print(global_tokenkvm.shape)
    dots_s=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # print(dots_s.shape)
    # _,_,dots_s=flash_attn_func(global_tokenq,global_tokenkvm[...,:d],0,return_attn_probs=True)
    # print(rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d').shape)
    # print(rearrange(global_tokenq, 'b h n s d -> (b n) s h d').shape)
    # out,sll,dots_s=flash_attn_func(rearrange(global_tokenq, 'b h n s d -> (b n) s h d'),rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d')[:,seg_len:],torch.zeros_like(rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d')[:,seg_len:],dtype=global_tokenq.dtype,device=global_tokenq.device),return_attn_probs=True)
    # print(dots_s)
    # print(out)
    # print(sll)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots_s,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*n1*h*nk,nk,device=global_tokenkvm.device,dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device=global_tokenkm.device,dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = s1*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n1 s ki) d -> b h n1 s (ki) d',ki=topklist,b=b,h=h,s=s)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    # if patchsize==1:
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]*(patchsize)+(patchsize-1)+indx
    # else:
    #     maskm = (q_mask[None, None, ..., None]) > maskm[..., None, :]*(patchsize)+(patchsize-1)+indx
    # print(maskm.shape)
    global_tokenkvm=global_tokenkvm.reshape(b,h,-1,2*d+1)[...,-seg_len:,:2*d]
    if global_tokenkvm.shape[-2]<seg_len:
        pad_offset = (0,) * (-1 - (-2)) * 2
        global_tokenkvm=F.pad(global_tokenkvm, (*pad_offset, seg_len-global_tokenkvm.shape[-2], 0), value = -1)
        # global_tokenkvm=pad_to_multiple(global_tokenkvm,seg_len,dim=-2,value=-1)
    

    return km,vm,maskm,global_tokenkvm


def Select_KV_C_mixc_kvpre1(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=32,kv_pre=None,indx=0):
    # b,h,n,s,d=shape
    # num_heads=h
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]
    num=seg_len//w
    (_, global_tokenq) = pad_to_multiple(global_tokenq, patchsize, dim = -2)
    global_tokenq=rearrange(global_tokenq, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenq=torch.mean(global_tokenq,dim=-2)

    
    # print(km.shape)
    # print(global_tokenkm.shape)
    b,h,n,s,w,d=shape
    # assert 
    # if nkv>seg_len:
    if n>=patchsize and nkv>seg_len:
        global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
        # (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device=global_tokenkvm.device,dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b = b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)

        # (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        
        global_tokenkvm = rearrange(global_tokenkvm, 'b h (n p) d -> b h n p d', p=seg_len)
        p=global_tokenkvm.shape[-2]
        global_tokenkvm = look_around(global_tokenkvm, **look_around_kwargs)
        if kv_pre is not None:
            # global_tokenkvm[...,0,:seg_len,:d*2]=torch.mean(rearrange(kv_pre, 'b h (n p) d -> b h n p d', p=patchsize),dim=-2)
            global_tokenkvm[...,0,:seg_len,:d*2]=kv_pre
    else:
        global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
        global_tokenq = global_tokenq[:,:,None]
        if kv_pre is not None:
            global_tokenkvm=torch.cat((kv_pre,global_tokenkvm),dim=-2)        
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device=global_tokenkvm.device,dtype=torch.int)-seg_len
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = global_tokenkvm[:,:,None]
        nkv=global_tokenkvm.shape[-2]
        

    # if n>=patchsize:
    #     global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
    # else:
    #     global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
    #     global_tokenq = global_tokenq[:,:,None]

    # print(global_tokenq.shape)
    # print(global_tokenkvm.shape)
    dots_s=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # print(dots_s.shape)
    # _,_,dots_s=flash_attn_func(global_tokenq,global_tokenkvm[...,:d],0,return_attn_probs=True)
    # print(rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d').shape)
    # print(rearrange(global_tokenq, 'b h n s d -> (b n) s h d').shape)
    # out,sll,dots_s=flash_attn_func(rearrange(global_tokenq, 'b h n s d -> (b n) s h d'),rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d')[:,seg_len:],torch.zeros_like(rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d')[:,seg_len:],dtype=global_tokenq.dtype,device=global_tokenq.device),return_attn_probs=True)
    # print(dots_s)
    # print(out)
    # print(sll)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots_s,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*n1*h*nk,nk,device=global_tokenkvm.device,dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device=global_tokenkm.device,dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = s1*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n1 s ki) d -> b h n1 s (ki) d',ki=topklist,b=b,h=h,s=s)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    # if patchsize==1:
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]*(patchsize)+(patchsize-1)+indx
    # else:
    #     maskm = (q_mask[None, None, ..., None]) > maskm[..., None, :]*(patchsize)+(patchsize-1)+indx
    # print(maskm.shape)
    global_tokenkvm=global_tokenkvm.reshape(b,h,-1,2*d+1)[...,-seg_len:,:2*d]
    if global_tokenkvm.shape[-2]<seg_len:
        pad_offset = (0,) * (-1 - (-2)) * 2
        global_tokenkvm=F.pad(global_tokenkvm, (*pad_offset, seg_len-global_tokenkvm.shape[-2], 0), value = -1)
        # global_tokenkvm=pad_to_multiple(global_tokenkvm,seg_len,dim=-2,value=-1)
    

    return km,vm,maskm,global_tokenkvm



class LlamaAttention_mss_ccc1(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

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
        autopad=True
        if autopad:
            orig_seq_len = hidden_states.shape[-2]
            (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)
        # query_r_len = query_r.shape[-2]
        # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        #select1
        attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # attn_outputs=attn_outputs[:,:query_r_len]
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        attn_output=torch.cat((attn_output,attn_outputs),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if needed_pad:
            attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class  LlamaAttention_mss_ccc_t(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=2048

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        attn_output1,attn_weights=forward_mss_local_inline_for_cccp(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2],topklist=[8],patchsize=[128],w=128)
        attn_output2,attn_weights=forward_mss_local_inline_for_sc(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3],topklist=[6],patchsize=[256],w=256)
        attn_output3,attn_weights=forward_mss_local_inline_for_sc(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:],topklist=[4],patchsize=[512],w=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc_semantic(self,query_r,key_r,value_r,seg_len=2048)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc_semantic(self,query_r,key_r,value_r)
        # forward_mss_local_inline_for_ccc_cross_semantic
        # # print(attn_output.shape,attn_output2.shape)
        # query_r_len = query_r.shape[-2]
        # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        #select1
        # print(key_r.shape)
        # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # if query_r.shape[-2]>self.seg_len*2:
        #     attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # else:
        #     attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # attn_outputs=attn_outputs[:,:query_r_len]
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        # attn_output=torch.cat((attn_output,attn_outputs),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
        # attn_output,attn_weights=forward_mss_local_t(self,query_states,key_states,value_states,topklist=[256,128],patchsize=[1,1],seg_len=16384)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaAttention_mss_ccc(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=2048

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)
        # query_r_len = query_r.shape[-2]
        # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        #select1
        # print(key_r.shape)
        # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc_p1(self,query_r,key_r,value_r,topklist=[256],seg_len=16384)
        attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r,topklist=[512])
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # if query_r.shape[-2]>self.seg_len*2:
        #     attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # else:
        #     attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # attn_outputs=attn_outputs[:,:query_r_len]
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        attn_output=torch.cat((attn_output,attn_outputs),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class LlamaAttention_mss_ccc2(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=2048

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)
        # query_r_len = query_r.shape[-2]
        # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        #select1
        # print(key_r.shape)
        # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        if query_r.shape[-2]>self.seg_len*2:
            attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r,topklist=[512])
        else:
            attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,topklist=[512],num_patch=256,seg_len=2048)
        # attn_outputs=attn_outputs[:,:query_r_len]
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        attn_output=torch.cat((attn_output,attn_outputs),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


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
    out = out.transpose(1, 2).contiguous()
    return out

def forward_localw_inline_fa(
    self,
    q, k, v,
    window_size = 256,
    k_pre=None,
    v_pre=None,
):
    shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, True, 0, window_size, True, 1, 0, False

    # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
    # (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
    # (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
    
    q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d'), (q, k, v))


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

    # bq = bq * scale

    mask_value = -torch.finfo(bk.dtype).max
    look_around_kwargs = dict(
        backward =  look_backward,
        forward =  look_forward,
        pad_value = pad_value
    )

    bk = look_around1(bk, **look_around_kwargs)
    bv = look_around1(bv, **look_around_kwargs)
    if k_pre is not None:
        # k_pre=rearrange(k_pre, 'b h (w n) d -> b h w n d', n = window_size)
        k_pre=rearrange(k_pre, 'b h n d -> (b h) n d',)
        v_pre=rearrange(v_pre, 'b h n d -> (b h) n d',)
        bk[:,0,:window_size]=k_pre
        bv[:,0,:window_size]=v_pre
    bq = rearrange(bq, '(b h) w n d -> (b w) n h d', h = shape[1])
    bk = rearrange(bk, '(b h) w n d -> (b w) n h d', h = shape[1])
    bv = rearrange(bv, '(b h) w n d -> (b w) n h d', h = shape[1])
    # if k_pre is not None:
    #     k_pre=rearrange(k_pre, 'b h (w n) d -> (b w) n h d', w = windows)
    #     v_pre=rearrange(v_pre, 'b h (w n) d -> (b w) n h d', w = windows)
    #     bk[0,:window_size]=k_pre
    #     bv[0,:window_size]=v_pre
    # print("bq",bq.shape)
    # print("bk",bk.shape)

    out=flash_attn_func(bq,bk,bv,softmax_scale=scale,causal=True)

    out = rearrange(out, '(b w) n h d -> b (w n) h d', w = windows)
    if autopad:
        out = out[:, :orig_seq_len, :]


    return out


def forward_mss_local_t(self,q,k,v,topklist=[128,96,64],patchsize=[1,4,16],local_windows=256,w=32,seg_len=2048):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
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

    qs=rearrange(q, 'b h (n s) d -> b h n s d',s=w)
    b,h,n,s,d=qs.shape
    num_heads=h
    global_tokenq=torch.mean(qs,dim=-2)

    backward=1

    Kt=[]
    Vt=[]
    Mask=[]

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    local_k,local_v,maskl=select_localw(self,kv=torch.cat((k,v),dim=-1),window_size=local_windows)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(maskl)

    k0,v0,mask0=Select_KV_C_mixc_w(self,qs.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_w(self,qs.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        # print('ki',k.shape)  
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # v = torch.cat((local_v, v0, v1, repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)), dim=-2)
    # v = torch.cat((v0, v1, ), dim=-2)
    # v = v0
    dotsl=rearrange(q, 'b h (n s) d -> b h n s d',s=local_windows)@local_k.transpose(-1,-2)*scaling
    dotsl.masked_fill_(~maskl, mask_value)
    del maskl

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # print('k',k.shape)

    dots=torch.matmul(qs,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    del kv_mask

    dots=torch.cat((rearrange(dotsl, 'b h n s t -> b h (n s) t'),rearrange(dots, 'b h n s t -> b h (n s) t')), dim=-1)

    attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)
    outl=rearrange(attn[...,:local_k.shape[-2]], 'b h (n s) t -> b h n s t',s=local_windows)@local_v
    outs=rearrange(attn[...,local_k.shape[-2]:], 'b h (n s) t -> b h n s t',s=w)@v

    out = rearrange(outl, 'b h n s d -> b h (n s) d')+rearrange(outs, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]

    out = out.transpose(1, 2).contiguous()


    return (out), attn



def forward_mss_local_inline_for_ccc(self,q,k,v,topklist=[128,96,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
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

    # if autopad:
    #     orig_seq_len = q.shape[2]
    #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    # k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    # v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    # b,h,n,s,d=q.shape
    # num_heads=h

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
    # b,h,ns,s2,d=q.shape
    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)
    # global_tokenq1=torch.cat((global_tokenq[...,:-1,:],global_tokenq[...,-1,:]),dim=-2)
    # global_tokenq1=torch.cat((rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,:-1,:],rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,-1,:][...,None,:]),dim=-2)
    # global_tokenq=(global_tokenq+global_tokenq1.view(global_tokenq.shape))/2


    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]
    # print("out",out.shape)
    # print(out.shape)
    # if autopad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn


def forward_mss_local_inline_for_ccc_p1(self,q,k,v,topklist=[128,96,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
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

    # if autopad:
    #     orig_seq_len = q.shape[2]
    #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    # k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    # v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    # b,h,n,s,d=q.shape
    # num_heads=h

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
    # b,h,ns,s2,d=q.shape
    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)
    # global_tokenq1=torch.cat((global_tokenq[...,:-1,:],global_tokenq[...,-1,:]),dim=-2)
    # global_tokenq1=torch.cat((rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,:-1,:],rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,-1,:][...,None,:]),dim=-2)
    # global_tokenq=(global_tokenq+global_tokenq1.view(global_tokenq.shape))/2


    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    # k=torch.cat(Kt, dim=-2)
    # v=torch.cat(Vt, dim=-2)
    # kv_mask=torch.cat(Mask, dim=-1).bool()

    k=k0
    v=v0
    kv_mask=mask0
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]
    # print("out",out.shape)
    # print(out.shape)
    # if autopad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn


def forward_mss_local_inline_for_ccc_cross(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
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

    # if autopad:
    #     orig_seq_len = q.shape[2]
    #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    # k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    # v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    # b,h,n,s,d=q.shape
    # num_heads=h

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
    # b,h,ns,s2,d=q.shape
    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)
    # global_tokenq1=torch.cat((global_tokenq[...,:-1,:],global_tokenq[...,-1,:]),dim=-2)
    global_tokenq1=torch.cat((rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,:-1,:],rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,-1,:][...,None,:]),dim=-2)
    global_tokenq=(global_tokenq+global_tokenq1.view(global_tokenq.shape))/2


    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device='cuda')
    # q_mask2=torch.arange(w,n*s*w+w,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask2=q_mask+w
    # q_mask2[-1,-1,:]=q_mask[-1,-1,:]
    # q_mask=torch.cat((q_mask,q_mask2),dim=-1)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_cross(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_cross(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # print("q",q.shape)
    # print("k",k.shape)
    # print("kv_mask",kv_mask.shape)
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]
    # print("out",out.shape)
    # print(out.shape)
    # if autopad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn

def forward_mss_local_inline_for_ccc_cross_semantic(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
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

    # if autopad:
    #     orig_seq_len = q.shape[2]
    #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    # k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    # v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    # b,h,n,s,d=q.shape
    # num_heads=h

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
    # b,h,ns,s2,d=q.shape
    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)
    # global_tokenq1=torch.cat((global_tokenq[...,:-1,:],global_tokenq[...,-1,:]),dim=-2)
    global_tokenq1=torch.cat((rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,:-1,:],rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,-1,:][...,None,:]),dim=-2)
    global_tokenq=(global_tokenq+global_tokenq1.view(global_tokenq.shape))/2
    global_tokenq=F.softmax(global_tokenq[...,None,:]@q.transpose(-1,-2)*scaling,dim=-1)@q
    global_tokenq=global_tokenq.reshape(b,h,n,-1,d)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device='cuda')
    # q_mask2=torch.arange(w,n*s*w+w,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask2=q_mask+w
    # q_mask2[-1,-1,:]=q_mask[-1,-1,:]
    # q_mask=torch.cat((q_mask,q_mask2),dim=-1)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_cross(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_cross(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # print("q",q.shape)
    # print("k",k.shape)
    # print("kv_mask",kv_mask.shape)
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]
    # print("out",out.shape)
    # print(out.shape)
    # if autopad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn

def forward_mss_local_inline_for_ccc_semantic(self,q,k,v,topklist=[128,96,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
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

    # if autopad:
    #     orig_seq_len = q.shape[2]
    #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    # k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    # v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    # b,h,n,s,d=q.shape
    # num_heads=h

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
    # b,h,ns,s2,d=q.shape
    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)
    global_tokenq=F.softmax(global_tokenq[...,None,:]@q.transpose(-1,-2)*scaling,dim=-1)@q
    global_tokenq=global_tokenq.reshape(b,h,n,-1,d)
    # global_tokenq1=torch.cat((global_tokenq[...,:-1,:],global_tokenq[...,-1,:]),dim=-2)
    # global_tokenq1=torch.cat((rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,:-1,:],rearrange(global_tokenq, 'b h n s d -> b h (n s) d')[...,-1,:][...,None,:]),dim=-2)
    # global_tokenq=(global_tokenq+global_tokenq1.view(global_tokenq.shape))/2


    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_w(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]
    # print("out",out.shape)
    # print(out.shape)
    # if autopad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn


def forward_mss_local_inline_for_ccc_R1(self,q,k,v,topklist=[64,64,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32,chunksize=16384):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True
    n=q.shape[-2]

    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    Pastkv=[]

    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (_,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device=k.device)
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    kv=torch.cat((k,v),dim=-1)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0,pastkv0=Select_KV_C_mixc_kvpre(self,q.shape,global_tokenq,q_mask,kv=kv,topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    # k0,v0,mask0=Select_KV_C_mixc_owntop(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    Pastkv.append(pastkv0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski,pastkvi=Select_KV_C_mixc_kvpre(self,q.shape,global_tokenq,q_mask,kv=kv,topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        # if chunksize>seg_len*patchsize[i]:
        #     ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=kv[...,-chunksize:,:],topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len,kv_pre=kv[...,-chunksize:])
        # else:
        #     ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=kv,topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        # ki,vi,maski=Select_KV_C_mixc_owntop(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        Pastkv.append(pastkvi)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    Pastkv=torch.cat(Pastkv, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    # print("out",out.shape)
    # print(out.shape)
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn, Pastkv


def forward_mss_local_inline_for_ccc_R2(self,q,k,v,topklist=[64,64,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32,chunksize=16384):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True
    n=q.shape[-2]

    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    Pastkv=[]

    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (_,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device=k.device)
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    kv=torch.cat((k,v),dim=-1)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0,pastkv0=Select_KV_C_mixc_kvpre(self,q.shape,global_tokenq,q_mask,kv=kv[...,seg_len*len(patchsize):,:],topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len,kv_pre=kv[...,:seg_len,:])
    # k0,v0,mask0=Select_KV_C_mixc_owntop(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    Pastkv.append(pastkv0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski,pastkvi=Select_KV_C_mixc_kvpre(self,q.shape,global_tokenq,q_mask,kv=kv[...,seg_len*len(patchsize):,:],topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len,kv_pre=kv[...,i*seg_len:(i+1)*seg_len,:])
        # if chunksize>seg_len*patchsize[i]:
        #     ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=kv[...,-chunksize:,:],topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len,kv_pre=kv[...,-chunksize:])
        # else:
        #     ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=kv,topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        # ki,vi,maski=Select_KV_C_mixc_owntop(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        Pastkv.append(pastkvi)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    Pastkv=torch.cat(Pastkv, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    # print("out",out.shape)
    # print(out.shape)
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn, Pastkv


class LlamaAttention_mss_ccc_R2(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384

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
        chunksize=self.chunksize
        selen=65536
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        group_size = int(q_len * group_size_ratio)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = selen // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

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
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_r = torch.cat([past_key_value[0], key_r], dim=2)
            value_r = torch.cat([past_key_value[1], value_r], dim=2)

        # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        query_l=query_states[:,:self.num_heads//2]
        key_l=key_states[:,:self.num_heads//2,-chunksize:]
        value_l=value_states[:,:self.num_heads//2,-chunksize:]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        attn_outputs=[]
        window_size=256
        attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256).contiguous()
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)

        #select1
        if past_key_value is None:
            attn_outputs,attn_weights,past_key_value1=forward_mss_local_inline_for_ccc_R1(self,query_r,key_r,value_r)
            past_key_value = (past_key_value1[...,:self.head_dim],past_key_value1[...,self.head_dim:]) if use_cache else None
        else:
            attn_outputs,attn_weights,past_key_value1=forward_mss_local_inline_for_ccc_R2(self,query_r,key_r,value_r)
            past_key_value = (past_key_value1[...,:self.head_dim],past_key_value1[...,self.head_dim:]) if use_cache else None
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_l,key_l,value_l)
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        attn_output=torch.cat((attn_output,attn_outputs),dim=-2)
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

class LlamaAttention_mss_ccc_Rt(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.trainingss=True

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
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

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
        if past_key_value is not None and ~self.trainingss:
            # reuse k, v, self_attention
            key_r = torch.cat([past_key_value[0], key_r], dim=2)
            value_r = torch.cat([past_key_value[1], value_r], dim=2)
        elif past_key_value is not None and self.trainingss:
            key_c=self.k_proj(past_key_value)
            value_c=self.v_proj(past_key_value)
            key_r = torch.cat([key_c, key_r], dim=2)
            value_r = torch.cat([value_c, value_r], dim=2)

        # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        query_l=query_states[:,:self.num_heads//2]
        key_l=key_states[:,:self.num_heads//2,-chunksize:]
        value_l=value_states[:,:self.num_heads//2,-chunksize:]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        attn_outputs=[]
        window_size=256
        min_value =-torch.finfo(key_l.dtype).max
        attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256).contiguous()
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)

        #select1
        if past_key_value is None:
            attn_outputs,attn_weights,past_key_value1=forward_mss_local_inline_for_ccc_R1(self,query_r,key_r,value_r)
            # past_key_value = (past_key_value1[...,:self.head_dim],past_key_value1[...,self.head_dim:]) if use_cache else None
        else:
            attn_outputs,attn_weights,past_key_value1=forward_mss_local_inline_for_ccc_R2(self,query_r,key_r,value_r)
            # past_key_value = (past_key_value1[...,:self.head_dim],past_key_value1[...,self.head_dim:]) if use_cache else None
        if use_cache and ~self.trainingss:
            past_key_value = (past_key_value1[...,:self.head_dim],past_key_value1[...,self.head_dim:])
        elif self.trainingss:
            past_hidden_stateskv=[]
            past_hidden_stateskv.append(hidden_states[...,-self.seg_len:,:])
            for i in range(1,len(patchsize)):
                temp=torch.mean(rearrange(hidden_states, 'b (n p) d -> b n p d', p=patchsize[i]),dim=-2)[...,-self.seg_len:,:]
                if temp.shape[-2]<self.seg_len:
                    temp[...,:self.seg_len-temp.shape[-2],:]=torch.ones_like(temp[...,:self.seg_len-temp.shape[-2],:],dtype=hidden_states.dtype,device=hidden_states.device)
                    temp[...,:self.seg_len-temp.shape[-2],:]=past_key_value[...,(i+1)*self.seg_len-(self.seg_len-temp.shape[-2]):(i+1)*self.seg_len,:]
                past_hidden_stateskv.append(temp)
            past_key_value = (torch.cat(past_hidden_stateskv,dim=-2))
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_l,key_l,value_l)
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        attn_output=torch.cat((attn_output,attn_outputs),dim=-2)
        # print(past_key_value)
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


class LlamaAttention_mss_ccc_Rm(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func=fit_func
        self.layer_idx=layer_idx


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==31:
                q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
            else:
                with torch.no_grad():
                    q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),q_his,past_key_value[1],causal=False)

            softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            # attn_output=(v_his+v_cur)/2

            softmax_lse_result = softmax_lse
            softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (query_states,attn_output,softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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

class LlamaAttention_mss_ccc_Rma(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==31:
                # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
                k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
                v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
            else:
                with torch.no_grad():
                    k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
                    v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1
            

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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


        # print("LlamaAttention_mss")


class LlamaAttention_mss_ccc_Rmam(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==31:
                # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
                k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
                v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
            else:
                with torch.no_grad():
                    k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
                    v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            # attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1
            attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                past_key_value = ((key_states.transpose(1, 2)/(self.chunk_num+1)+k_his*self.chunk_num/(self.chunk_num+1)),(value_states.transpose(1, 2)/(self.chunk_num+1)+v_hiss*self.chunk_num/(self.chunk_num+1)),softmax_lse_result)
                # past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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


        # print("LlamaAttention_mss")

class LlamaAttention_mss_ccc_Rmamm(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==31:
                # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
                k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
                v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
            else:
                with torch.no_grad():
                    k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
                    v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                past_key_value = ((attn_output),(attn_output),softmax_lse_result)
                # past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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


        # print("LlamaAttention_mss")

class LlamaAttention_mss_ccc_Rmax(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==31:
                # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
                k_his=self.fit_func_list[0]((past_key_value[0]))
                v_hiss=self.fit_func_list[1]((past_key_value[1]))
            else:
                with torch.no_grad():
                    k_his=self.fit_func_list[0]((past_key_value[0]))
                    v_hiss=self.fit_func_list[1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                # past_key_value = ((attn_output),(attn_output),softmax_lse_result)
                past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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


        # print("LlamaAttention_mss")

class LlamaAttention_mss_ccc_Rmaxx(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==31:
                # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
                k_his=self.fit_func_list[0]((past_key_value[0]))
                v_hiss=self.fit_func_list[1]((past_key_value[1]))
            else:
                with torch.no_grad():
                    k_his=self.fit_func_list[0]((past_key_value[0]))
                    v_hiss=self.fit_func_list[1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                # past_key_value = ((attn_output),(attn_output),softmax_lse_result)
                past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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


        # print("LlamaAttention_mss")



class LlamaAttention_mss_ccc_Rmc(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func=fit_func
        self.layer_idx=layer_idx
        self.chunk_num=1
        self.kcache_proj=nn.Sequential(
            nn.LayerNorm(self.head_dim*2),
            nn.Linear(self.head_dim*2,self.head_dim*4),
            nn.GELU(),
            nn.Linear(self.head_dim*4,self.head_dim*2),
        )
        self.vcache_proj=nn.Sequential(
            nn.LayerNorm(self.head_dim*2),
            nn.Linear(self.head_dim*2,self.head_dim*4),
            nn.GELU(),
            nn.Linear(self.head_dim*4,self.head_dim*2),
        )


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            # if self.layer_idx==31:
            #     q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
            # else:
            #     with torch.no_grad():
            #         q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            q_his=self.kvcache_proj(torch.cat((past_key_value[0],past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim*2)
            v_his=flash_attn_func(query_states.transpose(1, 2),past_key_value[1],past_key_value[1],causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his)/2+v_cur/2
            self.chunk_num=self.chunk_num+1

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=1
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (query_states,attn_output,softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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



class LlamaAttention_mss_ccc_Rmm(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func=fit_func
        self.layer_idx=layer_idx


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.layer_idx==0:
                q_his=self.fit_func(torch.cat((past_key_value[0],past_key_value[1]))).view(bsz, q_len, self.num_heads, self.head_dim)
            else:
                with torch.no_grad():
                    q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),q_his,past_key_value[1],causal=False)
            softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # print(past_key_value[2].shape)
            # print(softmax_lse_his.shape)
            # print(softmax_lse_his.transpose(1, 2)[...,None].shape)
            softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]
            softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            ratio_norm=(1/(1+torch.exp(softmax_lse.transpose(1, 2)-softmax_lse_his))).to(query_states.dtype)[...,None]
            attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)

            softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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


class LlamaAttention_mss_ccc_Rmb(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func=fit_func
        self.layer_idx=layer_idx
        self.kvcache_proj=nn.Sequential(
            nn.LayerNorm(self.head_dim*2),
            nn.Linear(self.head_dim*2,self.head_dim*4),
            nn.GELU(),
            nn.Linear(self.head_dim*4,self.head_dim*2),
        )
        self.kcache_proj=nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim,self.head_dim*2),
            nn.GELU(),
            nn.Linear(self.head_dim*2,self.head_dim),
        )
        self.vcache_proj=nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim,self.head_dim*2),
            nn.GELU(),
            nn.Linear(self.head_dim*2,self.head_dim),
        )




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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            # if self.layer_idx==0:
            #     q_his=self.fit_func(past_key_value[1].reshape(bsz, q_len, self.num_heads*self.head_dim)).view(bsz, q_len, self.num_heads, self.head_dim)
            # else:
            #     with torch.no_grad():
            #         q_his=self.fit_func(past_key_value[1].reshape(bsz, q_len, self.num_heads*self.head_dim)).view(bsz, q_len, self.num_heads, self.head_dim)
            # if self.layer_idx==31:
            #     q_his=self.fit_func(torch.cat((past_key_value[0],past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim*2)
            # else:
            #     with torch.no_grad():
            #         q_his=self.fit_func(torch.cat((past_key_value[0],past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim*2)

            # # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # v_his=flash_attn_func(query_states.transpose(1, 2),past_key_value[0],past_key_value[1],causal=False)
            k_his=self.kcache_proj((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=self.vcache_proj((past_key_value[1])).view(bsz, q_len, self.num_heads, self.head_dim)
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_his,causal=False)
            # q_his=self.kvcache_proj(torch.cat((past_key_value[0],past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim*2)
            # v_his=flash_attn_func(query_states.transpose(1, 2),q_his[...,:self.head_dim],q_his[...,self.head_dim:],causal=False)
            softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # print(past_key_value[2].shape)
            # print(softmax_lse_his.shape)
            # print(softmax_lse_his.transpose(1, 2)[...,None].shape)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse.transpose(1, 2)-softmax_lse_his))).to(query_states.dtype)[...,None]
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)

        

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

class LlamaAttention_mss_ccc_Rmd(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            # if self.layer_idx==31:
            #     # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
            #     # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
            #     k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
            #     v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
            # else:
            #     with torch.no_grad():
            #         k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
            #         v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            k_his=past_key_value[0]
            v_hiss=past_key_value[1]
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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

class LlamaAttention_mss_ccc_Rmd(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, fit_func_list=None):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.indx=0
        self.seg_len=2048
        self.chunksize=16384
        self.fit_func_list=fit_func_list
        self.layer_idx=layer_idx
        self.chunk_num=0


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
        # fit_func = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        chunksize=self.chunksize
        # chunksize=8192
        # chunksize=4096
        patchsize=self.patchsize
        bsz, q_len, _ = hidden_states.size()

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_chunk = q_len // chunksize

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
            # for i in range(num_chunk):
            #     query_states = self.q_proj(hidden_states)
            #     key_states = self.k_proj(hidden_states)
            #     value_states = self.v_proj(hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_cache=query_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if self.indx%num_chunk==0:
        #     past_key_value=None
        #     self.indx+=1
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # past_key_value1=(key_states, value_states)
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = past_key_value1 if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if past_key_value is not None:
            # reuse k, v, self_attention
            # if self.layer_idx==31:
            #     # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
            #     # k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0])).view(bsz, q_len, self.num_heads, self.head_dim)
            #     k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
            #     v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
            # else:
            #     with torch.no_grad():
            #         k_his=self.fit_func_list[self.chunk_num*2]((past_key_value[0]))
            #         v_hiss=self.fit_func_list[self.chunk_num*2+1]((past_key_value[1]))
                    # q_his=self.fit_func(torch.cat((past_key_value[0].transpose(1, 2),past_key_value[1]),dim=-1)).view(bsz, q_len, self.num_heads, self.head_dim)
                    # q_his=self.fit_func(past_key_value[1]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=self.fit_func(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            # q_his=(past_key_value[0]).view(bsz, q_len, self.num_heads, self.head_dim)
            k_his=past_key_value[0]
            v_hiss=past_key_value[1]
            v_his=flash_attn_func(query_states.transpose(1, 2),k_his,v_hiss,causal=False)

            # softmax_lse_his=torch.exp(past_key_value[2]-torch.max(past_key_value[2])).to(query_states.dtype)
            # softmax_lse_hisdim=torch.zeros_like(q_his,device=value_states.device,dtype=value_states.dtype)
            # softmax_lse_hisdim[...,0]=softmax_lse_his.transpose(1, 2)
            # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0].transpose(1, 2)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_his.transpose(1, 2)[...,None],causal=False).transpose(1, 2)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            # # softmax_lse_his=flash_attn_func(query_states.transpose(1, 2),q_his,softmax_lse_hisdim,causal=False)[...,0]

            # print(query_states.shape)
            # print(q_his.transpose(1, 2).shape)
            # print(past_key_value[2][...,None,:].shape)
            # softmax_lse_his=torch.logsumexp((torch.log(F.softmax(query_states@q_his.transpose(1, 2).transpose(-1, -2)*self.scaling,dim=-1)).to(query_states.dtype)+past_key_value[2][...,None,:]),dim=-1)
            # softmax_lse_his=(torch.log(softmax_lse_his)+torch.max(past_key_value[2])).to(query_states.dtype)
            

            v_cur,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)

            # ratio_norm=(1/(1+torch.exp(softmax_lse-softmax_lse_his))).to(query_states.dtype)[...,None].transpose(1, 2)
            # attn_output=ratio_norm*v_his+v_cur*(1-ratio_norm)
            attn_output=(v_his+v_cur)/2
            # attn_output=(v_his)*self.chunk_num/(self.chunk_num+1)+v_cur/(self.chunk_num+1)
            self.chunk_num=self.chunk_num+1

            softmax_lse_result = softmax_lse
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his[...,None], softmax_lse[...,None]],dim=-1), dim=-1)
            # softmax_lse_result = torch.logsumexp(torch.cat([softmax_lse_his.transpose(1, 2)[...,None], softmax_lse[...,None]],dim=-1), dim=-1).view(bsz, self.num_heads, q_len)
        else:
            self.chunk_num=0
            attn_output,softmax_lse,_=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True,return_attn_probs=True)


        if use_cache and past_key_value is not None:
            if past_key_value[0].shape[1]==key_states.shape[2]:
                past_key_value = ((key_states.transpose(1, 2)+k_his)/2,(value_states.transpose(1, 2)+v_hiss)/2,softmax_lse_result)
            # past_key_value = (query_states,attn_output,softmax_lse_result)
            # past_key_value = (query_states_cache,attn_output,softmax_lse_result)
        elif use_cache:
            past_key_value = (key_states.transpose(1, 2),value_states.transpose(1, 2),softmax_lse)
            # past_key_value = (query_states_cache,attn_output,softmax_lse)

        

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



def Select_KV_C_mixc_kvpre1(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=32,kv_pre=None,indx=0):
    # b,h,n,s,d=shape
    # num_heads=h
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]

    
    # print(km.shape)
    # print(global_tokenkm.shape)
    b,h,n,s,w,d=shape
    # assert 
    # if nkv>seg_len:
    if n>=patchsize and nkv>seg_len:
        global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
        # (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device=global_tokenkvm.device,dtype=torch.int)
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b = b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)

        # (_, global_tokenkvm) = pad_to_multiple(global_tokenkvm, seg_len, dim = -2)
        
        global_tokenkvm = rearrange(global_tokenkvm, 'b h (n p) d -> b h n p d', p=seg_len)
        p=global_tokenkvm.shape[-2]
        global_tokenkvm = look_around(global_tokenkvm, **look_around_kwargs)
        if kv_pre is not None:
            # global_tokenkvm[...,0,:seg_len,:d*2]=torch.mean(rearrange(kv_pre, 'b h (n p) d -> b h n p d', p=patchsize),dim=-2)
            global_tokenkvm[...,0,:seg_len,:d*2]=kv_pre
    else:
        global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
        global_tokenq = global_tokenq[:,:,None]
        if kv_pre is not None:
            global_tokenkvm=torch.cat((kv_pre,global_tokenkvm),dim=-2)        
        nk1=global_tokenkvm.shape[-2]
        offmask=torch.arange(0,nk1,1,device=global_tokenkvm.device,dtype=torch.int)-seg_len
        offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        global_tokenkvm = global_tokenkvm[:,:,None]
        nkv=global_tokenkvm.shape[-2]
        

    # if n>=patchsize:
    #     global_tokenq=rearrange(global_tokenq, 'b h (n p) s d -> b h n (p s) d', p=patchsize)
    # else:
    #     global_tokenq=rearrange(global_tokenq, 'b h n s d -> b h (n s) d')
    #     global_tokenq = global_tokenq[:,:,None]

    # print(global_tokenq.shape)
    # print(global_tokenkvm.shape)
    dots_s=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # print(dots_s.shape)
    # _,_,dots_s=flash_attn_func(global_tokenq,global_tokenkvm[...,:d],0,return_attn_probs=True)
    # print(rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d').shape)
    # print(rearrange(global_tokenq, 'b h n s d -> (b n) s h d').shape)
    # out,sll,dots_s=flash_attn_func(rearrange(global_tokenq, 'b h n s d -> (b n) s h d'),rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d')[:,seg_len:],torch.zeros_like(rearrange(global_tokenkvm[...,:d], 'b h n p d -> (b n) p h d')[:,seg_len:],dtype=global_tokenq.dtype,device=global_tokenq.device),return_attn_probs=True)
    # print(dots_s)
    # print(out)
    # print(sll)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots_s,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*n1*h*nk,nk,device=global_tokenkvm.device,dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device=global_tokenkm.device,dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = s1*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n1 s ki) d -> b h n1 s (ki) d',ki=topklist,b=b,h=h,s=s)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    # if patchsize==1:
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]*(patchsize)+(patchsize-1)+indx
    # else:
    #     maskm = (q_mask[None, None, ..., None]) > maskm[..., None, :]*(patchsize)+(patchsize-1)+indx
    # print(maskm.shape)
    global_tokenkvm=global_tokenkvm.reshape(b,h,-1,2*d+1)[...,-seg_len:,:2*d]
    if global_tokenkvm.shape[-2]<seg_len:
        pad_offset = (0,) * (-1 - (-2)) * 2
        global_tokenkvm=F.pad(global_tokenkvm, (*pad_offset, seg_len-global_tokenkvm.shape[-2], 0), value = -1)
        # global_tokenkvm=pad_to_multiple(global_tokenkvm,seg_len,dim=-2,value=-1)
    

    return km,vm,maskm,global_tokenkvm

def forward_mss_local_inline_for_sc1(self,q,k,v,topklist=[64,64,64],patchsize=[1,2,8],local_windows=128):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    seg_len=1024
    w=32
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
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        # orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (_,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device=k.device)
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    # print("out",out.shape)
    # print(out.shape)
    if autopad:
        out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn

def forward_mss_local_inline_for_sc(self,q,k,v,topklist=[128,96,64],patchsize=[1,2,8],num_patch=512,w=32,seg_len=1024):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=1024
    # w=32
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

    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    if autopad:
        orig_seq_len = q.shape[-2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        if orig_seq_len>=num_patch*w:
            (needed_pad,q) = pad_to_multiple_t(q, w, dim = -2)
            q = rearrange(q, 'b h (n w) d -> b h n w d', w=w)
        else:
            (needed_pad,q) = pad_to_multiple_t(q, num_patch, dim = -2)
            q = rearrange(q, 'b h (n p) d -> b h n p d', n=num_patch)

    b,h,n,s,d=q.shape
    global_tokenq=torch.mean(q,dim=-2)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device=k.device)
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_w3(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    # k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_w3(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    # out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    out = rearrange(out, 'b h n w d -> b h (n w) d')
    # print("out",out.shape)
    # print(out.shape)
    if needed_pad:
        out = out[..., -orig_seq_len:, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn

def forward_mss_local_inline_for_sc_semantic(self,q,k,v,topklist=[128,96,64],patchsize=[1,2,8],num_patch=512,w=32,seg_len=1024):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=1024
    # w=32
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

    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    if autopad:
        orig_seq_len = q.shape[-2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        if orig_seq_len>=num_patch*w:
            (needed_pad,q) = pad_to_multiple_t(q, w, dim = -2)
            q = rearrange(q, 'b h (n w) d -> b h n w d', w=w)
        else:
            (needed_pad,q) = pad_to_multiple_t(q, num_patch, dim = -2)
            q = rearrange(q, 'b h (n p) d -> b h n p d', n=num_patch)

    b,h,n,s,d=q.shape
    global_tokenq=torch.mean(q,dim=-2)
    # global_tokenq=F.softmax(global_tokenq[...,None,:]@q.transpose(-1,-2)*scaling,dim=-1)@q
    # global_tokenq=global_tokenq.reshape(b,h,n,-1,d)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device=k.device)
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_w3(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    # k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_w3(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    # out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    out = rearrange(out, 'b h n w d -> b h (n w) d')
    # print("out",out.shape)
    # print(out.shape)
    if needed_pad:
        out = out[..., -orig_seq_len:, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn


class LlamaAttention_mss_cc(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

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

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        # num_group = q_len // group_size

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
        attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
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



class LlamaAttention_mss_cc1(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

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

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        # num_group = q_len // group_size

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
        attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
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


def Select_KV_C_mixc_w(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=seg_len:
        # offmask=torch.arange(0,nk,1,device='cuda',dtype=torch.int)
        # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        # global_tokenkvm = global_tokenkvm[:,:,None]
        # nkv=global_tokenkvm.shape[-2]
        dots=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    else:
        mm=nkv//seg_len
        mmq=seg_len//(s*patchsize)
        rq=n-n//mmq*mmq
        # q1=rearrange(global_tokenq[...,:n-rq,:], 'b h (n p) d -> b h n p d', p=mmq)
        kv1=rearrange(global_tokenkvm[...,:nkv-rq*(s*patchsize),:d], 'b h (n p) d -> b h n p d', p=seg_len)
        q1=rearrange(global_tokenq[...,:n-rq,:], 'b h (n p) d -> b h n p d', n=kv1.shape[2])
        # kv1=rearrange(global_tokenkvm[...,:nkv-rq*(s*patchsize),:d], 'b h (n p) d -> b h n p d', n=q1.shape[2])
        kv1 = look_around(kv1, **look_around_kwargs)
        # print(q1.shape)
        # print(kv1.shape)
        dots=(q1@kv1.transpose(-1, -2))
        dots=rearrange(dots, 'b h n p d -> b h (n p) d')
        # kv1=rearrange(global_tokenkvm[...,n//mmq*seg_len-rq*(s*patchsize):-rq*(s*patchsize),:], 'b h (n p) d -> b h n p d', p=mmq)
        if rq>0:
            dots1=global_tokenq[...,n-rq:,:]@global_tokenkvm[...,-2*seg_len:,:d].transpose(-1, -2)
            dots=torch.cat((dots,dots1),dim=-2)

    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    if patchsize==1:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    else:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) > maskm[..., None, :]
    # print(maskm.shape)
    

    return km,vm,maskm

def forward_mss_local_inline_for_sc2(self,q,k,v,topklist=[64,64,64],patchsize=[1,2,8],num_patch=512,w=32,seg_len=1024):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=1024
    # w=32
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

    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # if autopad:
    #     orig_seq_len = q.shape[-2]
    #     # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
    #     if orig_seq_len>=num_patch*w:
    #         (needed_pad,q) = pad_to_multiple(q, w, dim = -2)
    #         q = rearrange(q, 'b h (n w) d -> b h n w d', w=w)
    #     else:
    #         (needed_pad,q) = pad_to_multiple(q, num_patch, dim = -2)
    #         q = rearrange(q, 'b h (n p) d -> b h n p d', n=num_patch)
    orig_seq_len = q.shape[-2]
    (needed_pad,q) = pad_to_multiple(q, 4, dim = -2)
    q = rearrange(q, 'b h (n w) d -> b h n w d', w=4)
    b,h,n,d=q.shape
    global_tokenq=torch.mean(q,dim=-2)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n,1,device=k.device)
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_w2(self,q.shape,q,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    # k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_w2(self,q.shape,q,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    

    k=torch.cat(Kt, dim=-2)
    v=torch.cat(Vt, dim=-2)
    kv_mask=torch.cat(Mask, dim=-1).bool()
    # print(k.shape)
    # print(kv_mask.shape)
    # patchscale=torch.cat(self.patchscale, dim=-1)

    dots=torch.matmul(q[...,None,:],k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask[...,None,:], mask_value)
    # print(dots.shape)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    # attn=rearrange(attn, 'b h n s t -> b h (n s) t')
    # print(v.shape)
    # print(attn.shape)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    # out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    out = rearrange(out, 'b h n w d -> b h (n w) d')
    # print("out",out.shape)
    # print(out.shape)
    # if needed_pad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn

def Select_KV_C_mixc_w2(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16):

    b,h,n,d=shape
    s=1
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    # global_tokenkvm=(kv)
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # if nkv<=seg_len:
    #     # offmask=torch.arange(0,nk,1,device='cuda',dtype=torch.int)
    #     # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    #     # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    #     # global_tokenkvm = global_tokenkvm[:,:,None]
    #     # nkv=global_tokenkvm.shape[-2]
    #     dots=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    # else:
        # mm=nkv//seg_len
        # mmq=seg_len//(s*patchsize)
        # rq=n-n//mmq*mmq
        # # q1=rearrange(global_tokenq[...,:n-rq,:], 'b h (n p) d -> b h n p d', p=mmq)
        # kv1=rearrange(global_tokenkvm[...,:nkv-rq*(s*patchsize),:d], 'b h (n p) d -> b h n p d', p=seg_len)
        # q1=rearrange(global_tokenq[...,:n-rq,:], 'b h (n p) d -> b h n p d', n=kv1.shape[2])
        # # kv1=rearrange(global_tokenkvm[...,:nkv-rq*(s*patchsize),:d], 'b h (n p) d -> b h n p d', n=q1.shape[2])
        # kv1 = look_around(kv1, **look_around_kwargs)
        # print(q1.shape)
        # print(kv1.shape)
        # dots=(q1@kv1.transpose(-1, -2))
        # dots=rearrange(dots, 'b h n p d -> b h (n p) d')
        # # kv1=rearrange(global_tokenkvm[...,n//mmq*seg_len-rq*(s*patchsize):-rq*(s*patchsize),:], 'b h (n p) d -> b h n p d', p=mmq)
        # if rq>0:
        #     dots1=global_tokenq[...,n-rq:,:]@global_tokenkvm[...,-2*seg_len:,:d].transpose(-1, -2)
        #     dots=torch.cat((dots,dots1),dim=-2)

    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    if patchsize==1:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., :]
        # maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    else:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) > maskm[..., :]
        # maskm = (q_mask[None, None, ..., None]//(patchsize)) > maskm[..., None, :]
    # print(maskm.shape)
    

    return km,vm,maskm



def Select_KV_C_mixc_w3(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    kvm=rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsize)
    global_tokenkvm=torch.mean(kvm,dim=-2)
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        if patchsize==1:
            maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
        else:
            maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]

        return km,vm,maskm
    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=seg_len:
        # offmask=torch.arange(0,nk,1,device='cuda',dtype=torch.int)
        # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        # global_tokenkvm = global_tokenkvm[:,:,None]
        # nkv=global_tokenkvm.shape[-2]
        dots=global_tokenq@global_tokenkvm[...,:d].transpose(-1, -2)
    else:
        mm=nkv//seg_len
        mmq=seg_len//(s*patchsize)
        rq=n-n//mmq*mmq
        # q1=rearrange(global_tokenq[...,:n-rq,:], 'b h (n p) d -> b h n p d', p=mmq)
        kv1=rearrange(global_tokenkvm[...,:nkv-rq*(s*patchsize),:d], 'b h (n p) d -> b h n p d', p=seg_len)
        q1=rearrange(global_tokenq[...,:n-rq,:], 'b h (n p) d -> b h n p d', n=kv1.shape[2])
        # kv1=rearrange(global_tokenkvm[...,:nkv-rq*(s*patchsize),:d], 'b h (n p) d -> b h n p d', n=q1.shape[2])
        kv1 = look_around(kv1, **look_around_kwargs)
        print(q1.shape)
        print(kv1.shape)
        dots=(q1@kv1.transpose(-1, -2))
        dots=rearrange(dots, 'b h n p d -> b h (n p) d')
        # kv1=rearrange(global_tokenkvm[...,n//mmq*seg_len-rq*(s*patchsize):-rq*(s*patchsize),:], 'b h (n p) d -> b h n p d', p=mmq)
        if rq>0:
            dots1=global_tokenq[...,n-rq:,:]@global_tokenkvm[...,-2*seg_len:,:d].transpose(-1, -2)
            dots=torch.cat((dots,dots1),dim=-2)

    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=global_tokenq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    if patchsize==1:
        maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    else:
        maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # print(maskm.shape)
    

    return km,vm,maskm


def select_localw(
    self,
    kv,
    window_size = 512
):

    # mask = default(mask, input_mask)

    # assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

    autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk =  True, -1, window_size, True, 1, 0, False

    # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
    # (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
    # (kv, packed_shape) = pack([kv], '* n d')

    # auto padding

    if autopad:
        orig_seq_len = kv.shape[1]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, window_size, dim = -2), (q, k, v))
        (needed_pad, kv) = pad_to_multiple(kv, window_size, dim = -2)

    b, h, n, dim_head, device, dtype = *kv.shape, kv.device, kv.dtype

    scale = dim_head ** -0.5

    assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

    windows = n // window_size

    seq = torch.arange(n, device = device)
    b_t = rearrange(seq, '(w n) -> 1 1 w n', w = windows, n = window_size)

    # bucketing

    # bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))
    bkv = rearrange(kv, 'b h (w n) d -> b h w n d', w = windows)


    look_around_kwargs = dict(
        backward =  look_backward,
        forward =  look_forward,
        pad_value = pad_value
    )

    bkv = look_around(bkv, **look_around_kwargs)
    # bv = look_around1(bv, **look_around_kwargs)

    # rotary embeddings

    # if exists(self.rel_pos):
    #     pos_emb, xpos_scale = self.rel_pos(bk)
    #     bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

    # calculate positions for masking

    bq_t = b_t
    bq_k = look_around(b_t, **look_around_kwargs)

    bq_t = rearrange(bq_t, '... i -> ... i 1')
    bq_k = rearrange(bq_k, '... j -> ... 1 j')

    pad_mask = bq_k != pad_value

    mask_value = -torch.finfo(bkv.dtype).max

    causal_mask = bq_t >= bq_k
    causal_mask = causal_mask&pad_mask
    d=kv.shape[-1]//2

    return bkv[...,:d],bkv[...,d:],causal_mask

# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()

#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#         # Build here to make `torch.jit.trace` work.
#         self._set_cos_sin_cache(
#             seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
#         )

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
#         self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         if seq_len > self.max_seq_len_cached:
#             self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#         )



# class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
#     """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
#         self.scaling_factor = scaling_factor
#         super().__init__(dim, max_position_embeddings, base, device)

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
#         t = t / self.scaling_factor

#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
#         self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin
    
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaAttention_mss_ccc_s(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=16384

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        #local
        # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp(self,query_r,key_r,value_r,topklist=[6],patchsize=[128],w=128)
        attn_outputs,attn_weights=forward_mss_local_inline_for_cccp(self,query_r,key_r,value_r,topklist=[5],patchsize=[512],w=512)
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # print(attn_output.shape,attn_output2.shape)
        # query_r_len = query_r.shape[-2]
        # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        #select1
        # print(key_r.shape)
        # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # attn_outputs,attn_weights=forward_localw_inline(self,query_r,key_r,value_r)
        # attn_outputs=attn_outputs[:,:query_r_len]
        # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # shift back
        attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaAttention_mss_ccc_sps(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, seg=16, top=256, merge=128):
        super().__init__()
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        self.seg = seg
        self.top = top
        self.merge = merge
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
        # self.patchscale = patchscale
        # self.patchscale1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        # self.attend=nn.Softmax(dim=-1)
        # self.dropout=nn.Dropout(0.)
        self.causal_mask=True
        # if patchscale[0]==0:
        
        self.patchscale=[]

        self.debug=0
        self.seg_len=16384

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

        # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # query_l=query_states[:,:self.num_heads//2]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        # #local
        # # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_s(self,query_r,key_r,value_r,topklist=[6],patchsize=[128],w=128)
        # # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # # print(attn_output.shape,attn_output2.shape)
        # # query_r_len = query_r.shape[-2]
        # # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        # #select1
        # # print(key_r.shape)
        # # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # # attn_outputs,attn_weights=forward_localw_inline(self,query_r,key_r,value_r)
        # # attn_outputs=attn_outputs[:,:query_r_len]
        # # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # # shift back
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[16],patchsize=[64],w=64,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_l,key_l,value_l,topklist=[12],patchsize=[128],w=128,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=4,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[8],w=8,mergep=16,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[64*4],patchsize=[16],w=16,mergep=64,flash_use=True)
        attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[self.top],patchsize=[self.seg],w=self.seg,mergep=self.merge,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=1,flash_use=True)
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Mistral_mss_ccc_sps2(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None, seg=16, top=256, merge=128, max_position_embeddings=32768, scaling_factor=4 ):
        super().__init__()
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        # print("LlamaAttention_mss")
        self.seg = seg
        self.top = top
        self.merge = merge
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
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=scaling_factor,
            base=self.rope_theta,
        )

        self.scaling=self.head_dim**(-0.5)
        self.max_seq = 16*1024
        self.local_windows = 128
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
        # self.patchscale = patchscale
        # self.patchscale1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        # self.attend=nn.Softmax(dim=-1)
        # self.dropout=nn.Dropout(0.)
        self.causal_mask=True
        # if patchscale[0]==0:
        
        self.patchscale=[]

        self.debug=0
        self.seg_len=16384

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_group = q_len // group_size

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

        # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # query_l=query_states[:,:self.num_heads//2]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        # #local
        # # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_s(self,query_r,key_r,value_r,topklist=[6],patchsize=[128],w=128)
        # # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # # print(attn_output.shape,attn_output2.shape)
        # # query_r_len = query_r.shape[-2]
        # # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        # #select1
        # # print(key_r.shape)
        # # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # # attn_outputs,attn_weights=forward_localw_inline(self,query_r,key_r,value_r)
        # # attn_outputs=attn_outputs[:,:query_r_len]
        # # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # # shift back
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[16],patchsize=[64],w=64,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_l,key_l,value_l,topklist=[12],patchsize=[128],w=128,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=4,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[8],w=8,mergep=16,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[64*4],patchsize=[16],w=16,mergep=64,flash_use=True)
        attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[self.top],patchsize=[self.seg],w=self.seg,mergep=self.merge,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=1,flash_use=True)
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaAttention_mss_ccc_sps_posorign(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
        # self.patchscale = patchscale
        # self.patchscale1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        # self.attend=nn.Softmax(dim=-1)
        # self.dropout=nn.Dropout(0.)
        self.causal_mask=True
        # if patchscale[0]==0:
        
        self.patchscale=[]
        # patchscale.append(1.0*32*torch.ones((self.local_windows*2)))
        # for i in range(len(self.topk)):
        #     patchscale.append(1.0*self.patchsize[i]*torch.ones((self.topk[i])))
        # # patchscale.append(1.0*64*torch.ones((self.max_seq//self.local_windows)))
        # patchscale=nn.Parameter(torch.cat(patchscale),requires_grad=True)
        # self.patchscale=nn.ParameterList()
        # self.patchscale.append(patchscale)
        # self.patchscale.append(nn.Parameter(1.0*64*torch.ones((self.max_seq//self.local_windows)),requires_grad=True))
        # # self.patchscale=[]
        # # self.patchscale.append(nn.Parameter(torch.tensor(1.0),requires_grad=True))
        # # self.patchscale.append(nn.Parameter(torch.tensor(4.0,requires_grad=True),requires_grad=True))
        # # self.patchscale.append(nn.Parameter(torch.tensor(64.0,requires_grad=True),requires_grad=True))
        # self.patchscale.append(nn.Parameter(1.0*torch.ones((self.topk[0])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=16384

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # query_l=query_states[:,:self.num_heads//2]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        # #local
        # # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_s(self,query_r,key_r,value_r,topklist=[6],patchsize=[128],w=128)
        # # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # # print(attn_output.shape,attn_output2.shape)
        # # query_r_len = query_r.shape[-2]
        # # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        # #select1
        # # print(key_r.shape)
        # # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # # attn_outputs,attn_weights=forward_localw_inline(self,query_r,key_r,value_r)
        # # attn_outputs=attn_outputs[:,:query_r_len]
        # # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # # shift back
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[16],patchsize=[64],w=64,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_l,key_l,value_l,topklist=[12],patchsize=[128],w=128,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=4,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[8],w=8,mergep=16,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[64*4],patchsize=[16],w=16,mergep=64,flash_use=True)
        attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[512],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=1,flash_use=True)
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class LlamaAttention_mss_ccc_spse(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
        # self.patchscale = patchscale
        # self.patchscale1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        # self.attend=nn.Softmax(dim=-1)
        # self.dropout=nn.Dropout(0.)
        self.causal_mask=True
        # if patchscale[0]==0:
        
        self.patchscale=[]
        # patchscale.append(1.0*32*torch.ones((self.local_windows*2)))
        # for i in range(len(self.topk)):
        #     patchscale.append(1.0*self.patchsize[i]*torch.ones((self.topk[i])))
        # # patchscale.append(1.0*64*torch.ones((self.max_seq//self.local_windows)))
        # patchscale=nn.Parameter(torch.cat(patchscale),requires_grad=True)
        # self.patchscale=nn.ParameterList()
        # self.patchscale.append(patchscale)
        # self.patchscale.append(nn.Parameter(1.0*64*torch.ones((self.max_seq//self.local_windows)),requires_grad=True))
        # # self.patchscale=[]
        # # self.patchscale.append(nn.Parameter(torch.tensor(1.0),requires_grad=True))
        # # self.patchscale.append(nn.Parameter(torch.tensor(4.0,requires_grad=True),requires_grad=True))
        # # self.patchscale.append(nn.Parameter(torch.tensor(64.0,requires_grad=True),requires_grad=True))
        # self.patchscale.append(nn.Parameter(1.0*torch.ones((self.topk[0])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=16384

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        # query_states=query_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states=key_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states=value_states.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, self.num_heads, self.head_dim).transpose(1, 2)
        # query_l=query_states[:,:self.num_heads//2]
        # key_l=key_states[:,:self.num_heads//2]
        # value_l=value_states[:,:self.num_heads//2]


        # attn_weights = torch.matmul(query_l, key_l.transpose(2, 3)) / math.sqrt(self.head_dim)

        # # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        # #local
        # # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_s(self,query_r,key_r,value_r,topklist=[6],patchsize=[128],w=128)
        # # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # # print(attn_output.shape,attn_output2.shape)
        # # query_r_len = query_r.shape[-2]
        # # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        # #select1
        # # print(key_r.shape)
        # # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # # attn_outputs,attn_weights=forward_localw_inline(self,query_r,key_r,value_r)
        # # attn_outputs=attn_outputs[:,:query_r_len]
        # # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # # shift back
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[16],patchsize=[64],w=64,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_l,key_l,value_l,topklist=[12],patchsize=[128],w=128,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=4,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[8],w=8,mergep=16,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[64*4],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[512],patchsize=[16],w=16,mergep=128,flash_use=True)
        attn_output=flash_attn_func(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2),causal=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=1,flash_use=True)
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, None, past_key_value

def forward_mss_local_inline_for_cccs(self,q,k,v,topklist=[128,96,64],patchsize=[1,2,8],local_windows=128,seg_len=2048,w=32):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
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

    # if autopad:
    #     orig_seq_len = q.shape[2]
    #     (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))

    # q=rearrange(q, 'b h (n s) d -> b h n s d',n=local_windows)
    # k=rearrange(k, 'b h (n s) d -> b h n s d',n=local_windows)
    # v=rearrange(v, 'b h (n s) d -> b h n s d',n=local_windows)
    # b,h,n,s,d=q.shape
    # num_heads=h

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
    # b,h,ns,s2,d=q.shape
    # global_tokenq, global_tokenk, global_tokenv=torch.mean(q,dim=-2),torch.mean(k,dim=-2),torch.mean(v,dim=-2)
    # k=rearrange(k, 'b h n s d -> b h (n s) d')
    # v=rearrange(v, 'b h n s d -> b h (n s) d')
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]

    # dotsl=torch.matmul(q,local_k.transpose(-1, -2))*scaling
    # local_mask=torch.ones((s,(backward+1)*s)).cuda()
    # # local_mask=torch.tril(local_mask, diagonal=s)-torch.tril(local_mask, diagonal=-1)
    # local_mask=torch.tril(local_mask, diagonal=backward*s)
    # maskl=local_mask[None, None, None, :].bool()
    # dotsl.masked_fill_(~maskl, mask_value)
    # del maskl

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, seg_len, dim = -2)

    # q = rearrange(q, 'b h n s d -> b h (n s) d')
    # q = rearrange(q, 'b h (n p) d -> b (h n) p d', p=seg_len)
    # q = rearrange(q, 'b h (s w) d -> b h s w d', w=w)
    # b,hs,ns,w,d=q.shape
    q = rearrange(q, 'b h (n p) d -> b h n p d', p=seg_len)
    q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,w,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h)
    # Kt.append(local_k)
    # Vt.append(local_v)
    # Mask.append(repeat(local_mask[None, None, None, :, :], '1 1 1 s m -> b h n s m', n=n,b=b,h=h))

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s*w,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s w) -> n s w',n=n,s=s,w=w)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    # g_mask=torch.arange(0,n,1).cuda()
    # maskg=(q_mask[:, :, None]//s) > g_mask[:]
    # # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling

    # # Kt.append(repeat(global_tokenk[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Vt.append(repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n))
    # # Mask.append(repeat(maskg[None, None, :, :], '1 1 n s m -> b h n s m', b=b,h=h))
    # dotsg=torch.matmul(q,global_tokenk[...,None,:,:].transpose(-1, -2))*scaling
    # dotsg.masked_fill_(~maskg, mask_value)
    # del maskg
    # # Mask.append(maskg[None, None, :, :])
    
    if len(topklist)>1:
        k=torch.cat(Kt, dim=-2)
        v=torch.cat(Vt, dim=-2)
        kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)
    else:
        k=k0
        v=v0
        kv_mask=mask0.bool()


    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s w d -> b h (n s w) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]
    # print("out",out.shape)
    # print(out.shape)
    # if autopad:
    #     out = out[..., :orig_seq_len, :]
    # out = (self.out_proj(out))
    # print(out.shape)
    out = out.transpose(1, 2).contiguous()


    return (out), attn

def Select_KV_C_mixc_patchsize(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,semantic=True):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=selq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm



def Select_KV_C_mixc_patchsize_s(self,shape,q,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,semantic=True):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    scaling=d**(-0.5)
    selq=torch.mean(q,dim=-2)
    selq=F.softmax(selq[...,None,:]@q.transpose(-1,-2)*scaling,dim=-1)@q
    selq=selq.reshape(b,h,n,d)

    selk1=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk1,dim=-2)
    selk=F.softmax(selk[...,None,:]@selk1.transpose(-1,-2)*scaling,dim=-1)@selk1
    selk=selk.reshape(b,h,n,d)

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    # print(dots_s.shape)
    # print(topklist)
    
    selectm = torch.topk(dots,k=int(topklist),dim=-1)[-1]

    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    s1=selq.shape[-2]
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm




def forward_mss_local_inline_for_cccp(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    n=q.shape[-2]

    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_patchsize(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_patchsize(self,q.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    
    if len(topklist)>1:
        k=torch.cat(Kt, dim=-2)
        v=torch.cat(Vt, dim=-2)
        kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)
    else:
        k=k0
        v=v0
        kv_mask=mask0.bool()


    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]

    out = out.transpose(1, 2).contiguous()


    return (out), attn


def forward_mss_local_inline_for_cccp_s(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128):
# def forward_mss_local_inline_for1(self,q,k,v,topklist=[128,64,64],patchsize=[1,2,8]):
    # bsz, q_len, _ = q.size()
    # topklist=[256,128,128]
    # patchsize=[1,2,8]
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]
    # local_windows=128
    # seg_len=2048
    # w=32
    scaling=self.head_dim**(-0.5)
    causal_mask=True
    # print("LlamaAttention_mss")
    autopad=True

    n=q.shape[-2]

    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    local_k = look_around(k, **look_around_kwargs)
    local_v = look_around(v, **look_around_kwargs)
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    k0,v0,mask0=Select_KV_C_mixc_patchsize_s(self,q.shape,q,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    Kt.append(k0)
    Vt.append(v0)
    Mask.append(mask0)
    indx=topklist[0]*patchsize[0]

    for i in range(1,len(topklist)):
        # print(i,topklist[i])
        ki,vi,maski=Select_KV_C_mixc_patchsize_s(self,q.shape,q,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist[i],patchsize=patchsize[i],seg_len=seg_len)
        Kt.append(ki)
        Vt.append(vi)
        Mask.append(maski)
        indx+=topklist[i]*patchsize[i]
    
    
    if len(topklist)>1:
        k=torch.cat(Kt, dim=-2)
        v=torch.cat(Vt, dim=-2)
        kv_mask=torch.cat(Mask, dim=-1).bool()
    # patchscale=torch.cat(self.patchscale, dim=-1)
    else:
        k=k0
        v=v0
        kv_mask=mask0.bool()


    dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    dots.masked_fill_(~kv_mask, mask_value)
    # print(kv_mask[0,0,:2,:2,:].int())
    # print(kv_mask[0,0,-2:,-2:,:].int())
    del kv_mask
    # quit()
    # dots = rearrange(dots, 'b (h n) ns w t -> b h n s t')
    # attn=nn.functional.softmax(dots+torch.log(self.patchscale), dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0]),dotsg+torch.log(self.patchscale[-1])),dim=-1)
    # dots=torch.cat((dotsl,dots+torch.log(self.patchscale[0])),dim=-1)
    # v=torch.cat((local_v,v,repeat(global_tokenv[...,None,:,:], 'b h 1 s d -> b h n s d', n=n)),dim=-2)
    # print("v",v.shape)
    attn=nn.functional.softmax(dots, dim=-1)
    out=torch.matmul(attn,v)

    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # out = rearrange(out, 'b (h n) ns w d -> b h (n ns w) d',h=h)
    # print(out.shape)
    out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:]

    out = out.transpose(1, 2).contiguous()


    return (out), attn


def forward_mss_local_inline_for_cccp_d(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)

    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n=q.shape[2]
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
        q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
        k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
        v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
        # print(qf.shape,kf.shape,vf.shape)
        outfront=flash_attn_func(qf,kf,vf,causal=True)
        # print(q0.shape,k0.shape,v0.shape)
        outback=flash_attn_func(q0,k0,v0,causal=True)
        outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-topklist[0])
    else:
        kf = k[...,:topklist[0]*w,:]
        vf = v[...,:topklist[0]*w,:]
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        q0 = q[:,:,topklist[0]:]
        k0 = k0
        v0 = v0
        maskfront=torch.tril(torch.ones((topklist[0]*w,topklist[0]*w),dtype=k.dtype,device=k.device)).bool()
        dotsfront=qf@kf.transpose(-1, -2)*scaling
        dotsfront.masked_fill_(~maskfront, mask_value)
        attnfront=nn.functional.softmax(dotsfront, dim=-1)
        outfront=torch.matmul(attnfront,vf)
        outfront = rearrange(outfront, 'b h n d -> b n h d')

        dotsback=q0@k0.transpose(-1, -2)*scaling
        dotsback.masked_fill_(~mask0, mask_value)
        attnback=nn.functional.softmax(dotsback, dim=-1)
        outback=torch.matmul(attnback,v0)
        outback = rearrange(outback, 'b h n s d -> b (n s) h d')
    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn


def forward_mss_local_inline_for_cccp_dt(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topklist[0]*patchsize[0]:
        k0,v0,mask0=Select_KV_C_mixc_patchsize_dflashattn(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)

    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n=q.shape[2]
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>topklist[0]*patchsize[0]:
            q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
            k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
            v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
            outback=flash_attn_func(q0,k0,v0,causal=True)
            outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-topklist[0])
            out = torch.cat((outfront,outback),dim=1)
    else:
        kf = k[...,:topklist[0]*w,:]
        vf = v[...,:topklist[0]*w,:]
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        dotsfront=qf@kf.transpose(-1, -2)*scaling
        dotsfront.masked_fill_(~maskfront, mask_value)
        attnfront=nn.functional.softmax(dotsfront, dim=-1)
        outfront=torch.matmul(attnfront,vf)
        outfront = rearrange(outfront, 'b h n d -> b n h d')
        out=outfront

        if k.shape[-2]>topklist[0]*patchsize[0]:
            q0 = q[:,:,topklist[0]:]
            k0 = k0
            v0 = v0
            dotsback=q0@k0.transpose(-1, -2)*scaling
            dotsback.masked_fill_(~mask0, mask_value)
            attnback=nn.functional.softmax(dotsback, dim=-1)
            outback=torch.matmul(attnback,v0)
            outback = rearrange(outback, 'b h n s d -> b (n s) h d')
            out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn


def forward_mss_local_inline_for_cccp_dtt(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topklist[0]*patchsize[0]:
        k0,v0,mask0=Select_KV_C_mixc_patchsize_dflashattn(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)

    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n=q.shape[2]
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>topklist[0]*patchsize[0]:
            if orig_seq_len%patchsize[0]==0:
                q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k0[:,:,:], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v0[:,:,:], 'b h n tk d -> (b n) tk h d')
                
                outback=flash_attn_func(q0,k0,v0,causal=True)
                outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-topklist[0])
                out = torch.cat((outfront,outback),dim=1)
            else:
                q0 = rearrange(q[:,:,topklist[0]:-1], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k0[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v0[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                if k0.shape[0]>0:
                    outback=flash_attn_func(q0,k0,v0,causal=True)
                    outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-topklist[0]-1)
                    out = torch.cat((outfront,outback),dim=1)

                if orig_seq_len>(n-1)*s:
                    qr=rearrange(q[:,:,-1][:,:,:orig_seq_len-(n-1)*s], 'b h n d -> b n h d')
                    kr=rearrange(k, 'b h n d -> b n h d')
                    vr=rearrange(v, 'b h n d -> b n h d')
                    outr=flash_attn_func(qr,kr,vr,causal=True)
                    if k0.shape[0]>0:
                        out = torch.cat((outfront,outback,outr),dim=1)
                    else:
                        out = torch.cat((outfront,outr),dim=1)
    else:
        kf = k[...,:topklist[0]*w,:]
        vf = v[...,:topklist[0]*w,:]
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        dotsfront=qf@kf.transpose(-1, -2)*scaling
        dotsfront.masked_fill_(~maskfront, mask_value)
        attnfront=nn.functional.softmax(dotsfront, dim=-1)
        outfront=torch.matmul(attnfront,vf)
        outfront = rearrange(outfront, 'b h n d -> b n h d')
        out=outfront

        if k.shape[-2]>topklist[0]*patchsize[0]:
            q0 = q[:,:,topklist[0]:]
            k0 = k0
            v0 = v0
            dotsback=q0@k0.transpose(-1, -2)*scaling
            dotsback.masked_fill_(~mask0, mask_value)
            attnback=nn.functional.softmax(dotsback, dim=-1)
            outback=torch.matmul(attnback,v0)
            outback = rearrange(outback, 'b h n s d -> b (n s) h d')
            out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn


def forward_mss_local_inline_for_cccp_dtt_seg(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,flash_use=False,seg=4096):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    # q_mask=torch.arange(0,n*s,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    ddd=seg//w

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topklist[0]*patchsize[0]:
        k0,v0,mask0=Select_KV_C_mixc_patchsize_dflashattn(self,q.shape,global_tokenq[...,ddd:,:],None,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)

    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n=q.shape[2]
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:seg,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:seg,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:ddd], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>seg:
            if orig_seq_len%patchsize[0]==0:
                q0 = rearrange(q[:,:,ddd:], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k0[:,:,:], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v0[:,:,:], 'b h n tk d -> (b n) tk h d')
                
                outback=flash_attn_func(q0,k0,v0,causal=True)
                outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-ddd)
                out = torch.cat((outfront,outback),dim=1)
            else:
                q0 = rearrange(q[:,:,ddd:-1], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k0[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v0[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                if k0.shape[0]>0:
                    outback=flash_attn_func(q0,k0,v0,causal=True)
                    outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-ddd-1)
                    out = torch.cat((outfront,outback),dim=1)

                if orig_seq_len>(n-1)*s:
                    qr=rearrange(q[:,:,-1][:,:,:orig_seq_len-(n-1)*s], 'b h n d -> b n h d')
                    kr=rearrange(k, 'b h n d -> b n h d')
                    vr=rearrange(v, 'b h n d -> b n h d')
                    outr=flash_attn_func(qr,kr,vr,causal=True)
                    if k0.shape[0]>0:
                        out = torch.cat((outfront,outback,outr),dim=1)
                    else:
                        out = torch.cat((outfront,outr),dim=1)
    # else:
    #     kf = k[...,:topklist[0]*w,:]
    #     vf = v[...,:topklist[0]*w,:]
    #     qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
    #     maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
    #     dotsfront=qf@kf.transpose(-1, -2)*scaling
    #     dotsfront.masked_fill_(~maskfront, mask_value)
    #     attnfront=nn.functional.softmax(dotsfront, dim=-1)
    #     outfront=torch.matmul(attnfront,vf)
    #     outfront = rearrange(outfront, 'b h n d -> b n h d')
    #     out=outfront

    #     if k.shape[-2]>topklist[0]*patchsize[0]:
    #         q0 = q[:,:,topklist[0]:]
    #         k0 = k0
    #         v0 = v0
    #         dotsback=q0@k0.transpose(-1, -2)*scaling
    #         dotsback.masked_fill_(~mask0, mask_value)
    #         attnback=nn.functional.softmax(dotsback, dim=-1)
    #         outback=torch.matmul(attnback,v0)
    #         outback = rearrange(outback, 'b h n s d -> b (n s) h d')
    #         out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn



def forward_mss_local_inline_for_cccp_dttt(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topklist[0]*patchsize[0]:
        k0,v0,mask0=Select_KV_C_mixc_patchsize_dflashattn(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)

    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n=q.shape[2]
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>topklist[0]*patchsize[0]:
            if not needed_pad:
                q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k0[:,:,:], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v0[:,:,:], 'b h n tk d -> (b n) tk h d')
                
                outback=flash_attn_func(q0,k0,v0,causal=True)
                outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-topklist[0])
                out = torch.cat((outfront,outback),dim=1)
            else:
                q0 = rearrange(q[:,:,topklist[0]:-1], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k0[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v0[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                if k0.shape[0]>0:
                    outback=flash_attn_func(q0,k0,v0,causal=True)
                    outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-topklist[0]-1)
                    out = torch.cat((outfront,outback),dim=1)

                if orig_seq_len>(n-1)*s:
                    qr=rearrange(q[:,:,-1][:,:,:orig_seq_len-(n-1)*s], 'b h n d -> b n h d')
                    kr=rearrange(k, 'b h n d -> b n h d')
                    vr=rearrange(v, 'b h n d -> b n h d')
                    outr=flash_attn_func(qr,kr,vr,causal=True)
                    if k0.shape[0]>0:
                        out = torch.cat((outfront,outback,outr),dim=1)
                    else:
                        out = torch.cat((outfront,outr),dim=1)
    else:
        kf = k[...,:topklist[0]*w,:]
        vf = v[...,:topklist[0]*w,:]
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        dotsfront=qf@kf.transpose(-1, -2)*scaling
        dotsfront.masked_fill_(~maskfront, mask_value)
        attnfront=nn.functional.softmax(dotsfront, dim=-1)
        outfront=torch.matmul(attnfront,vf)
        outfront = rearrange(outfront, 'b h n d -> b n h d')
        out=outfront

        if k.shape[-2]>topklist[0]*patchsize[0]:
            q0 = q[:,:,topklist[0]:]
            k0 = k0
            v0 = v0
            dotsback=q0@k0.transpose(-1, -2)*scaling
            dotsback.masked_fill_(~mask0, mask_value)
            attnback=nn.functional.softmax(dotsback, dim=-1)
            outback=torch.matmul(attnback,v0)
            outback = rearrange(outback, 'b h n s d -> b (n s) h d')
            out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn


def forward_mss_local_inline_for_cccp_dttm(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,mergep=4,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    # scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w*mergep, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    # q_mask=torch.arange(0,n*s,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topklist[0]*patchsize[0]:
        k00,v00,mask0=Select_KV_C_mixc_patchsize_dflashattn_merge(self,q.shape,global_tokenq[...,((topklist[0])):,:],None,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len,mergep=mergep)
        # k00,v00,mask0=Select_KV_C_mixc_patchsize_dflashattn_merge(self,q.shape,global_tokenq[...,((topklist[0]-1)//mergep+1)*mergep:,:],None,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len,mergep=mergep)

    # q = rearrange(q, 'b h (n m) p d -> b h n (m p) d', m=mergep)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # topklist[0]=((topklist[0]-1)//mergep+1)*mergep
    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n1=q.shape[2]//mergep
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>topklist[0]*patchsize[0]:
            q00 = rearrange(q[:,:,topklist[0]:], 'b h (n m) p d -> b h n (m p) d',m=mergep)
            n=q00.shape[2]
            s=q00.shape[3]
            if not needed_pad:
                q0 = rearrange(q00, 'b h n p d -> (b n) p h d')
                # print("q0",q0.shape)
                k0 = rearrange(k00, 'b h n tk d -> (b n) tk h d')
                # print("k0",k0.shape)
                v0 = rearrange(v00, 'b h n tk d -> (b n) tk h d')

                
                outback=flash_attn_func(q0,k0,v0,causal=True)
                # print("outback",outback.shape)
                outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n)
                out = torch.cat((outfront,outback),dim=1)
            else:
                # q0 = rearrange(q[:,:,topklist[0]:-1], 'b h n p d -> (b n) p h d')
                q0 = rearrange(q00[:,:,:-1], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k00[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v00[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                if k0.shape[0]>0:
                    outback=flash_attn_func(q0,k0,v0,causal=True)
                    outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-1)
                    out = torch.cat((outfront,outback),dim=1)

                if orig_seq_len>(n1-1)*s:
                    qr=rearrange(q00[:,:,-1][:,:,:orig_seq_len-(n1-1)*s], 'b h n d -> b n h d')
                    kr=rearrange(k, 'b h n d -> b n h d')
                    vr=rearrange(v, 'b h n d -> b n h d')
                    # kr=rearrange(k00[:,:,-1][:,:,:-(-orig_seq_len+(n)*s)], 'b h n d -> b n h d')
                    # vr=rearrange(v00[:,:,-1][:,:,:-(-orig_seq_len+(n)*s)], 'b h n d -> b n h d')
                    outr=flash_attn_func(qr,kr,vr,causal=True)
                    if k0.shape[0]>0:
                        out = torch.cat((outfront,outback,outr),dim=1)
                    else:
                        out = torch.cat((outfront,outr),dim=1)
    # else:
    #     kf = k[...,:topklist[0]*w,:]
    #     vf = v[...,:topklist[0]*w,:]
    #     qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
    #     maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
    #     dotsfront=qf@kf.transpose(-1, -2)*scaling
    #     dotsfront.masked_fill_(~maskfront, mask_value)
    #     attnfront=nn.functional.softmax(dotsfront, dim=-1)
    #     outfront=torch.matmul(attnfront,vf)
    #     outfront = rearrange(outfront, 'b h n d -> b n h d')
    #     out=outfront

    #     if k.shape[-2]>topklist[0]*patchsize[0]:
    #         q0 = q[:,:,topklist[0]:]
    #         k0 = k0
    #         v0 = v0
    #         dotsback=q0@k0.transpose(-1, -2)*scaling
    #         dotsback.masked_fill_(~mask0, mask_value)
    #         attnback=nn.functional.softmax(dotsback, dim=-1)
    #         outback=torch.matmul(attnback,v0)
    #         outback = rearrange(outback, 'b h n s d -> b (n s) h d')
    #         out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # print(out.shape)
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    # print(out.shape)
    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn

def forward_mss_local_inline_for_cccp_dttm2(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,mergep=4,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    Kt=[]
    Vt=[]
    Mask=[]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad,q) = pad_to_multiple(q, w*mergep, dim = -2)

    q = rearrange(q, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # print(q.shape)
    global_tokenq=torch.mean(q,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    # q_mask=torch.arange(0,n*s,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topklist[0]*patchsize[0]:
        k00,v00,mask0=Select_KV_C_mixc_patchsize_dflashattn_merge2(self,q.shape,global_tokenq[...,((topklist[0])):,:],None,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len,mergep=mergep)
        # k00,v00,mask0=Select_KV_C_mixc_patchsize_dflashattn_merge(self,q.shape,global_tokenq[...,((topklist[0]-1)//mergep+1)*mergep:,:],None,kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len,mergep=mergep)

    # q = rearrange(q, 'b h (n m) p d -> b h n (m p) d', m=mergep)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=q.shape
    # topklist[0]=((topklist[0]-1)//mergep+1)*mergep
    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n1=q.shape[2]//mergep
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>topklist[0]*patchsize[0]:
            q00 = rearrange(q[:,:,topklist[0]:], 'b h (n m) p d -> b h n (m p) d',m=mergep)
            n=q00.shape[2]
            s=q00.shape[3]
            if not needed_pad:
                q0 = rearrange(q00, 'b h n p d -> (b n) p h d')
                # print("q0",q0.shape)
                k0 = rearrange(k00, 'b h n tk d -> (b n) tk h d')
                # print("k0",k0.shape)
                v0 = rearrange(v00, 'b h n tk d -> (b n) tk h d')

                
                outback=flash_attn_func(q0,k0,v0,causal=True)
                # print("outback",outback.shape)
                outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n)
                out = torch.cat((outfront,outback),dim=1)
            else:
                # q0 = rearrange(q[:,:,topklist[0]:-1], 'b h n p d -> (b n) p h d')
                q0 = rearrange(q00[:,:,:-1], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k00[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v00[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                if k0.shape[0]>0:
                    outback=flash_attn_func(q0,k0,v0,causal=True)
                    outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-1)
                    out = torch.cat((outfront,outback),dim=1)

                if orig_seq_len>(n1-1)*s:
                    qr=rearrange(q00[:,:,-1][:,:,:orig_seq_len-(n1-1)*s], 'b h n d -> b n h d')
                    kr=rearrange(k, 'b h n d -> b n h d')
                    vr=rearrange(v, 'b h n d -> b n h d')
                    # kr=rearrange(k00[:,:,-1][:,:,:-(-orig_seq_len+(n)*s)], 'b h n d -> b n h d')
                    # vr=rearrange(v00[:,:,-1][:,:,:-(-orig_seq_len+(n)*s)], 'b h n d -> b n h d')
                    outr=flash_attn_func(qr,kr,vr,causal=True)
                    if k0.shape[0]>0:
                        out = torch.cat((outfront,outback,outr),dim=1)
                    else:
                        out = torch.cat((outfront,outr),dim=1)
    # else:
    #     kf = k[...,:topklist[0]*w,:]
    #     vf = v[...,:topklist[0]*w,:]
    #     qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
    #     maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
    #     dotsfront=qf@kf.transpose(-1, -2)*scaling
    #     dotsfront.masked_fill_(~maskfront, mask_value)
    #     attnfront=nn.functional.softmax(dotsfront, dim=-1)
    #     outfront=torch.matmul(attnfront,vf)
    #     outfront = rearrange(outfront, 'b h n d -> b n h d')
    #     out=outfront

    #     if k.shape[-2]>topklist[0]*patchsize[0]:
    #         q0 = q[:,:,topklist[0]:]
    #         k0 = k0
    #         v0 = v0
    #         dotsback=q0@k0.transpose(-1, -2)*scaling
    #         dotsback.masked_fill_(~mask0, mask_value)
    #         attnback=nn.functional.softmax(dotsback, dim=-1)
    #         outback=torch.matmul(attnback,v0)
    #         outback = rearrange(outback, 'b h n s d -> b (n s) h d')
    #         out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # print(out.shape)
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    # print(out.shape)
    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn



def forward_mss_local_inline_for_cccp_dttms(self,q,k,v,topklist=[4,4],patchsize=[128,256],local_windows=128,seg_len=2048,w=128,mergep=4,flash_use=False):
    patchscale=[torch.tensor(1.0),torch.tensor(3.2),torch.tensor(64.0)]

    scaling=self.head_dim**(-0.5)
    causal_mask=True
    autopad=True

    n=q.shape[-2]

    backward=1
    
    mask_value=-torch.finfo(k.dtype).max

    # topsum=int(torch.sum(torch.tensor(topklist)*torch.tensor(patchsize)//patchsize[0]))
    topsum=int(torch.sum(torch.tensor(topklist)*torch.tensor(patchsize)))
    Kt=[]
    Vt=[]
    Mask=[]
    qs=q[...,topsum:,:]
    if autopad:
        orig_seq_len = q.shape[2]
        # (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, local_windows, dim = -2), (q, k, v))
        (needed_pad, qs) = pad_to_multiple(qs, w*mergep, dim = -2)

    qs = rearrange(qs, 'b h (n p) d -> b h n p d', p=w)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=qs.shape
    # print(q.shape)
    global_tokenq=torch.mean(qs,dim=-2)

    mask_value=-torch.finfo(k.dtype).max
    q_mask=torch.arange(0,n*s,1,device='cuda')
    q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # q_mask=torch.arange(0,ns*w,1,device='cuda')
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=ns,s=w)
    # q_mask=torch.arange(0,n*s,1).cuda()
    # q_mask = rearrange(q_mask, '(n s) -> n s',n=n,s=s)
    # topsum=int(torch.sum(torch.tensor(topklist)*torch.tensor(patchsize)))

    # print(q.shape)
    # k0,v0,mask0=Select_KV_C_mixc_patchsize_d(self,q.shape,global_tokenq[...,topklist[0]:,:],q_mask[topklist[0]:,:],kv=torch.cat((k,v),dim=-1),topklist=topklist[0],patchsize=patchsize[0],seg_len=seg_len)
    if k.shape[-2]>topsum:
        k00,v00,mask0=Select_KV_C_mixc_patchsize_dflashattn_mergems(self,qs.shape,global_tokenq,q_mask,kv=torch.cat((k,v),dim=-1),topklist=topklist,patchsizelist=patchsize,seg_len=seg_len,mergep=mergep)

    # q = rearrange(q, 'b h (n m) p d -> b h n (m p) d', m=mergep)
    # q = rearrange(q, 'b h n (s w) d -> b h n s w d', w=w)
    b,h,n,s,d=qs.shape
    # dots=torch.matmul(q,k.transpose(-1, -2))*scaling
    # dots.masked_fill_(~kv_mask, mask_value)
    # del kv_mask
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v)

    # dots=torch.matmul(q,k0.transpose(-1, -2))*scaling
    # dots.masked_fill_(~mask0, mask_value)
    # del mask0
    # attn=nn.functional.softmax(dots, dim=-1)
    # out=torch.matmul(attn,v0)

    # print(k0.shape,v0.shape)
    n1=qs.shape[2]//mergep
    # kf = rearrange(k[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # vf = rearrange(v[...,:topklist[0]*w,:], 'b h n d -> b n h d')
    # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b (n p) h d')
    # q0 = rearrange(q[:,:,topklist[0]:], 'b h n p d -> (b n) p h d')
    # k0 = rearrange(k0, 'b h n tk d -> (b n) tk h d')
    # v0 = rearrange(v0, 'b h n tk d -> (b n) tk h d')
    if flash_use:
        kf = rearrange(k[...,:topsum,:], 'b h n d -> b n h d')
        vf = rearrange(v[...,:topsum,:], 'b h n d -> b n h d')
        qf = rearrange(q[:,:,:topsum], 'b h n d -> b n h d')
        # qf = rearrange(q[:,:,:topsum], 'b h n p d -> b (n p) h d')
        outfront=flash_attn_func(qf[:,:kf.shape[1]],kf,vf,causal=True)
        out=outfront


        # kf = k[...,:topklist[0]*w,:]
        # vf = v[...,:topklist[0]*w,:]
        # qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
        # maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
        # dotsfront=qf@kf.transpose(-1, -2)*scaling
        # dotsfront.masked_fill_(~maskfront, mask_value)
        # attnfront=nn.functional.softmax(dotsfront, dim=-1)
        # outfront=torch.matmul(attnfront,vf)
        # outfront = rearrange(outfront, 'b h n d -> b n h d')
        # print(q0.shape,k0.shape,v0.shape)
        if k.shape[-2]>topsum:
            q00 = rearrange(qs, 'b h (n m) p d -> b h n (m p) d',m=mergep)
            n=q00.shape[2]
            s=q00.shape[3]
            if not needed_pad:
                q0 = rearrange(q00, 'b h n p d -> (b n) p h d')
                # print("q0",q0.shape)
                k0 = rearrange(k00, 'b h n tk d -> (b n) tk h d')
                # print("k0",k0.shape)
                v0 = rearrange(v00, 'b h n tk d -> (b n) tk h d')

                
                outback=flash_attn_func(q0,k0,v0,causal=True)
                # print("outback",outback.shape)
                outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n)
                out = torch.cat((outfront,outback),dim=1)
            else:
                # q0 = rearrange(q[:,:,topklist[0]:-1], 'b h n p d -> (b n) p h d')
                q0 = rearrange(q00[:,:,:-1], 'b h n p d -> (b n) p h d')
                k0 = rearrange(k00[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                v0 = rearrange(v00[:,:,:-1], 'b h n tk d -> (b n) tk h d')
                if k0.shape[0]>0:
                    outback=flash_attn_func(q0,k0,v0,causal=True)
                    outback = rearrange(outback, '(b n) s h d -> b (n s) h d',n=n-1)
                    out = torch.cat((outfront,outback),dim=1)

                if orig_seq_len>(n1-1)*s:
                    qr=rearrange(q00[:,:,-1][:,:,:orig_seq_len-(n1-1)*s-topsum], 'b h n d -> b n h d')
                    kr=rearrange(k, 'b h n d -> b n h d')
                    vr=rearrange(v, 'b h n d -> b n h d')
                    # kr=rearrange(k00[:,:,-1][:,:,:-(-orig_seq_len+(n)*s)], 'b h n d -> b n h d')
                    # vr=rearrange(v00[:,:,-1][:,:,:-(-orig_seq_len+(n)*s)], 'b h n d -> b n h d')
                    outr=flash_attn_func(qr,kr,vr,causal=True)
                    if k0.shape[0]>0:
                        out = torch.cat((outfront,outback,outr),dim=1)
                    else:
                        out = torch.cat((outfront,outr),dim=1)
    # else:
    #     kf = k[...,:topklist[0]*w,:]
    #     vf = v[...,:topklist[0]*w,:]
    #     qf = rearrange(q[:,:,:topklist[0]], 'b h n p d -> b h (n p) d')
        
    #     maskfront=torch.tril(torch.ones((qf.shape[-2],kf.shape[-2]),dtype=k.dtype,device=k.device)).bool()
    #     dotsfront=qf@kf.transpose(-1, -2)*scaling
    #     dotsfront.masked_fill_(~maskfront, mask_value)
    #     attnfront=nn.functional.softmax(dotsfront, dim=-1)
    #     outfront=torch.matmul(attnfront,vf)
    #     outfront = rearrange(outfront, 'b h n d -> b n h d')
    #     out=outfront

    #     if k.shape[-2]>topklist[0]*patchsize[0]:
    #         q0 = q[:,:,topklist[0]:]
    #         k0 = k0
    #         v0 = v0
    #         dotsback=q0@k0.transpose(-1, -2)*scaling
    #         dotsback.masked_fill_(~mask0, mask_value)
    #         attnback=nn.functional.softmax(dotsback, dim=-1)
    #         outback=torch.matmul(attnback,v0)
    #         outback = rearrange(outback, 'b h n s d -> b (n s) h d')
    #         out = torch.cat((outfront,outback),dim=1)

    # print(qf.shape,kf.shape,vf.shape)
    # print(outfront.shape)
    # print(outback.shape)


    # out = torch.cat((outfront,outback),dim=1)
    # out = outback
    
    # out = rearrange(out, 'b h n s d -> b h (n s) d')
    # print(out.shape)
    if needed_pad:
        out=out[...,:orig_seq_len,:,:]

    # print(out.shape)
    out = out.contiguous()
    # out = out.transpose(1, 2).contiguous()
    attn=None


    return (out), attn


def Select_KV_C_mixc_patchsize_d1(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize and False:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-2).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    # print(dots_s.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],dtype=selectm.dtype,device=selectm.device)

    selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)

    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device='cuda',dtype=torch.int)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc_patchsize_d(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize and False:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    # print(dots.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
        # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

    # print(selt)
    selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)
    # print(selectm.shape)
    # print(selectm)


    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    maskm=kvm[...,-1]
    vm = kvm[...,d:-1]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    # maskm = None
    maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc_patchsize_dflashattn(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    # offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize and False:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    # print(dots.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
        # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

    # print(selt)
    selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)
    # print(selectm.shape)
    # print(selectm)


    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    # maskm=kvm[...,-1]
    vm = kvm[...,d:]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = None
    # maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc_patchsize_dflashattn_ms(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    # offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize and False:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    # print(dots.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
        # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

    # print(selt)
    selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)
    # print(selectm.shape)
    # print(selectm)


    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    # maskm=kvm[...,-1]
    vm = kvm[...,d:]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = None
    # maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm


def Select_KV_C_mixc_patchsize_dflashattn_merge(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None,mergep=4):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize*mergep, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    # offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize and False:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-mergep).bool()
    # masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    del masksel
    # print(dots.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topklist-mergep),dim=-1)[-1]
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = rearrange(selectm, 'b h (n m) t -> b h n t m', m=mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = rearrange(selectm, 'b h n t m -> b h n (t m)', m=mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = extract_unique_elements(selectm=selectm,k=topklist-mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
        # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

    # print(selt)
    # selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    selt= repeat(selt[None,None,:], '1 1 (nk m)-> b h nk m', b=b,h=h,m=mergep)
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)
    # print(selectm.shape)
    # print(selectm)


    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]//mergep
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)
    # print("offm",offm.shape)
    # print("selectm",selectm.shape)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    # maskm=kvm[...,-1]
    vm = kvm[...,d:]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = None
    # maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc_patchsize_dflashattn_merge2(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None,mergep=4):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize*mergep, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    # offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    if topklist>mergep:
        # selq=global_tokenq
        selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
        selk=torch.mean(selk,dim=-2)
        masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-mergep).bool()
        # masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

        # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
        dots=selq@selk.transpose(-1,-2)
        min_value=-torch.finfo(selk.dtype).max
        dots.masked_fill_(~masksel, min_value)
        del masksel
        # print(dots.shape)
        # print(topklist)
        selectm = torch.topk(dots,k=int(topklist-mergep),dim=-1)[-1]
        # print(selectm[0,0,:mergep,:4*mergep])
        selectm = rearrange(selectm, 'b h (n m) t -> b h n t m', m=mergep)
        # print(selectm[0,0,:mergep,:4*mergep])
        selectm = rearrange(selectm, 'b h n t m -> b h n (t m)', m=mergep)
        # print(selectm[0,0,:mergep,:4*mergep])
        selectm = extract_unique_elements(selectm=selectm,k=topklist-mergep)
        # print(selectm[0,0,:mergep,:4*mergep])
        
        if start is not None:
            selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
        else:
            selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
            # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

        # print(selt)
        # selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
        selt= repeat(selt[None,None,:], '1 1 (nk m)-> b h nk m', b=b,h=h,m=mergep)
        # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
        # print(selectm.shape)
        # print(selt.shape)
        selectm =torch.cat((selectm,selt),dim=-1)
    else:
        selt=torch.arange(kv.shape[-2]//patchsize-selq.shape[-2],kv.shape[-2]//patchsize,1,dtype=torch.int,device=kv.device)
        selectm= repeat(selt[None,None,:], '1 1 (nk m)-> b h nk m', b=b,h=h,m=mergep)


    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]//mergep
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)
    # print("offm",offm.shape)
    # print("selectm",selectm.shape)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    # maskm=kvm[...,-1]
    vm = kvm[...,d:]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = None
    # maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm

def Select_KV_C_mixc_patchsize_dflashattn_merge_sen(self,shape,global_tokenq,q_mask,kv,topklist,patchsize,seg_len=2048,w=16,start=None,mergep=4):

    b,h,n,s,d=shape
    backward=1

    look_around_kwargs = dict(
        backward =  backward,
        forward =  0,
        pad_value = -1
    )

    shfitp=kv.shape[-2]%patchsize

    (_, kv) = pad_to_multiple_t(kv, patchsize*mergep, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)
    global_tokenkvm=kv
    nkv=global_tokenkvm.shape[-2]
    # offmask=torch.arange(0,nkv,1,device='cuda',dtype=torch.int)
    # offmask= repeat(offmask[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    if nkv<=topklist*patchsize and False:
        km=global_tokenkvm[...,:d]
        maskm=global_tokenkvm[...,-1]
        vm = global_tokenkvm[...,d:-1]
        maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]

        return km,vm,maskm
    
    global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    senmatic_q=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    senmatic_k=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((senmatic_q.shape[-2],senmatic_k.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-mergep).bool()
    # masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=senmatic_q@senmatic_k.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    del masksel
    # print(dots.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topklist-mergep),dim=-1)[-1]
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = rearrange(selectm, 'b h (n m) t -> b h n t m', m=mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = rearrange(selectm, 'b h n t m -> b h n (t m)', m=mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = extract_unique_elements(selectm=selectm,k=topklist-mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-senmatic_q.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
        # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

    # print(selt)
    # selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    selt= repeat(selt[None,None,:], '1 1 (nk m)-> b h nk m', b=b,h=h,m=mergep)
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)
    # print(selectm.shape)
    # print(selectm)


    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=senmatic_q.shape[-2]//mergep
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist)
    # print("offm",offm.shape)
    # print("selectm",selectm.shape)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist,b=b,h=h,p=patchsize)
    km=kvm[...,:d]
    # maskm=kvm[...,-1]
    vm = kvm[...,d:]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    maskm = None
    # maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return km,vm,maskm



def Select_KV_C_mixc_patchsize_dflashattn_mergems(self,shape,global_tokenq,q_mask,kv,topklist,patchsizelist,seg_len=2048,w=16,start=None,mergep=4):

    b,h,n,s,d=shape
    backward=1
    topk=topklist[0]
    patchsize=patchsizelist[0]
    topsum=int(torch.sum(torch.tensor(topklist)*torch.tensor(patchsizelist)//patchsize))

    (_, kv) = pad_to_multiple_t(kv, patchsizelist[-1]*mergep, dim = -2)
    # (_, q) = pad_to_multiple_t(q, patchsize, dim = -2)
    # kvm=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    # global_tokenkvm=torch.mean(kvm,dim=-2)


    # selq=rearrange(q, 'b h (n p) d -> b h n p d', p=patchsize)
    # selq=torch.mean(selq,dim=-2)
    selq=global_tokenq
    selk=rearrange(kv[...,:d], 'b h (n p) d -> b h n p d', p=patchsize)
    selk=torch.mean(selk,dim=-2)
    masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topsum-mergep).bool()
    # masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=selk.shape[-2]-selq.shape[-2]-mergep).bool()
    # masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-mergep).bool()
    # masksel=torch.tril(torch.ones((selq.shape[-2],selk.shape[-2]),dtype=selk.dtype,device=selk.device),diagonal=topklist-1).bool()

    # global_tokenkvm=torch.cat((global_tokenkvm,offmask),dim=-1)
    dots=selq@selk.transpose(-1,-2)
    min_value=-torch.finfo(selk.dtype).max
    dots.masked_fill_(~masksel, min_value)
    # print(dots.shape)
    # print(topklist)
    selectm = torch.topk(dots,k=int(topsum-mergep),dim=-1)[-1]
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = rearrange(selectm, 'b h (n m) t -> b h n t m', m=mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    selectm = rearrange(selectm, 'b h n t m -> b h n (t m)', m=mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    selectt = extract_unique_elements(selectm=selectm,k=topsum-mergep)
    # print(selectm[0,0,:mergep,:4*mergep])
    
    if start is not None:
        selt=torch.arange(start,selk.shape[-2],dtype=selectm.dtype,device=selectm.device)
    else:
        selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],1,dtype=selectm.dtype,device=selectm.device)
        # selt=torch.arange(selk.shape[-2]-selq.shape[-2],selk.shape[-2],device='cuda',dtype=torch.int)

    # print(selt)
    # selt= repeat(selt[None,None,:], '1 1 nk -> b h nk', b=b,h=h)[...,None]
    selectm=selectt[...,:topklist[0]-mergep]
    selt= repeat(selt[None,None,:], '1 1 (nk m)-> b h nk m', b=b,h=h,m=mergep)
    # selectm = torch.topk(dots,k=int(topklist-1),dim=-1)[-1]
    # print(selectm.shape)
    # print(selt.shape)
    selectm =torch.cat((selectm,selt),dim=-1)
    # print(selectm.shape)
    # print(selectm)
    klist=[]
    vlist=[]
    global_tokenkvm=rearrange(kv, 'b h (n p) d -> b h n (p d)', p=patchsize)
    nk=global_tokenkvm.shape[-2]
    # n1=global_tokenkvm.shape[2]
    offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
    # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
    n=selq.shape[-2]//mergep
    # print(n)
    offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[0])
    # print("offm",offm.shape)
    # print("selectm",selectm.shape)

    kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

    # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
    kvm0 = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist[0],b=b,h=h,p=patchsize)
    # klist.append(kvm[...,:d])
    # vlist.append(kvm[...,d:])
    # km=kvm[...,:d]
    # # maskm=kvm[...,-1]
    # vm = kvm[...,d:]
    end=topklist[0]
    for i in range(1,len(topklist)):
        start=end
        end=start+topklist[i]*patchsizelist[i]//patchsize
        # print("start:",start)
        # print("end:",end)
        selectm=selectt[...,start:end]
        selectm = extract_unique_elements(selectm=selectm//(patchsizelist[i]//patchsize),k=topklist[i])

        global_tokenkvm=torch.mean(rearrange(kv, 'b h (n p) d -> b h n p d', p=patchsizelist[i]//patchsize),dim=-2)
        nkv=global_tokenkvm.shape[-2]
        global_tokenkvm=rearrange(global_tokenkvm, 'b h (n p) d -> b h n (p d)', p=patchsize)
        nk=global_tokenkvm.shape[-2]
        # n1=global_tokenkvm.shape[2]
        offm=torch.arange(0,b*h*nk,nk,device=selectm.device,dtype=selectm.dtype)
        # offm=torch.arange(0,b*nk*num_heads,nk,device='cuda',dtype=torch.int)
        n=selq.shape[-2]//mergep
        # print(n)
        offm= repeat(offm[:,None], 'd 1 -> (d b)', b = n*topklist[i])
        # print("offm",offm.shape)
        # print("selectm",selectm.shape)

        kvm=torch.index_select(global_tokenkvm.reshape(-1,global_tokenkvm.shape[-1]),dim=0,index=selectm.reshape(-1)+offm)

        # kvm = rearrange(kvm, '(b h n ki) d -> b h n (ki) d',ki=topklist,b=b,h=h)
        kvm = rearrange(kvm, '(b h n ki) (p d) -> b h n (ki p) d',ki=topklist[i],b=b,h=h,p=patchsize)
        klist.append(kvm[...,:d])
        vlist.append(kvm[...,d:])
        # km=kvm[...,:d]
        # # maskm=kvm[...,-1]
        # vm = kvm[...,d:]
    # km = rearrange(km, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # vm = rearrange(vm, '(b h n ns ki) d -> b h n ns (ki) d',ki=topklist,b=b,h=num_heads,ns=ns)
    # print(vm.shape)
    # print(q_mask[None, None, ..., None].shape)
    # print(maskm[..., None, :].shape)
    klist.append(kvm0[...,:d])
    vlist.append(kvm0[...,d:])
    klist=torch.cat(klist,dim=-2)
    vlist=torch.cat(vlist,dim=-2)
    maskm = None
    # maskm = (q_mask[None, None, ..., None]) >= maskm[..., None, :]
    # if patchsize==1:
    #     maskm = (q_mask[None, None, ..., None]//(patchsize)) >= maskm[..., None, :]
    # else:
    #     maskm = ((q_mask[None, None, ..., None]+shfitp)//(patchsize)) > maskm[..., None, :]
    # # print(maskm.shape)
    

    return klist,vlist,maskm

class LlamaAttention_mss_ccc_sps_mix(nn.Module):
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
        self.topk = [128,64,64,64]
        self.topk = [64,64,64,64]
        self.topk = [64,64,64]
        # self.topk = [64,64,64,64,64]
        # self.topk = (320,160,128)
        self.patchsize = (1,4,16)
        self.patchsize = [1,2,8]
        patchsize=[1,2,4,8,16]
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
        # self.patchscale.append(nn.Parameter(4.0*torch.ones((self.topk[1])),requires_grad=True))
        # self.patchscale.append(nn.Parameter(64.0*torch.ones((self.topk[2])),requires_grad=True))

        self.debug=0
        self.seg_len=16384

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
        # autopad=True
        # if autopad:
        #     orig_seq_len = hidden_states.shape[-2]
        #     (needed_pad, hidden_states)= pad_to_multiple(hidden_states, 16384, dim = -2)
        bsz, q_len, _ = hidden_states.size()


        group_size = int(q_len * 1/8)

        # if q_len % group_size > 0:
        #     raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
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

        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

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

        # # attn_output=forward_localw_inline(self,query_states[:,:self.num_heads//4],key_states[:,:self.num_heads//4],value_states[:,:self.num_heads//4],window_size=512)
        # #local
        # # attn_output=forward_localw_inline(self,query_states,key_states,value_states,window_size=512)
        # # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_output=forward_localw_inline_fa(self,query_l,key_l,value_l,window_size=256)
        # attn_output=forward_localw_inline(self,query_l,key_l,value_l,window_size=512)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_s(self,query_r,key_r,value_r,topklist=[6],patchsize=[128],w=128)
        # # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.num_heads//2, self.head_dim)

        # # attn_output1,attn_weights=forward_mss_inline_own1(self,query_states[:,self.num_heads//4:self.num_heads//2],key_states[:,self.num_heads//4:self.num_heads//2],value_states[:,self.num_heads//4:self.num_heads//2])
        # # attn_output2,attn_weights=forward_mss_inline_own2(self,query_states[:,self.num_heads//2:self.num_heads//4*3],key_states[:,self.num_heads//2:self.num_heads//4*3],value_states[:,self.num_heads//2:self.num_heads//4*3])
        # # attn_output3,attn_weights=forward_mss_inline_own(self,query_states[:,self.num_heads//4*3:],key_states[:,self.num_heads//4*3:],value_states[:,self.num_heads//4*3:])
        # # # print(attn_output.shape,attn_output2.shape)
        # # query_r_len = query_r.shape[-2]
        # # (needed_pad, query_r), (_, key_r), (_, value_r) = map(lambda t: pad_to_multiple(t, self.patchsize[-1], dim = -2), (query_r, key_r, value_r))

        # #select1
        # # print(key_r.shape)
        # # attn_outputs=forward_localw_inline(self,query_l,key_l,value_l,window_size=256)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_ccc(self,query_r,key_r,value_r)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048)
        # # attn_outputs,attn_weights=forward_mss_local_inline_for_sc(self,query_r,key_r,value_r,num_patch=256,seg_len=2048,patchsize=[1,2,8])
        # # attn_outputs,attn_weights=forward_localw_inline(self,query_r,key_r,value_r)
        # # attn_outputs=attn_outputs[:,:query_r_len]
        # # attn_output,attn_weights=forward_mss_local_inline_for_sc(self,query_states,key_states,value_states)
        # # attn_output,attn_weights=forward_mss_local_inline_for_s(self,query_states,key_states,value_states,self.topk,patchsize=self.patchsize)
        # # attn_output,attn_weights=forward_mss_inline_ownj1(self,query_states,key_states,value_states)
        # # attn_output=torch.cat((attn_output,attn_output1,attn_output2,attn_output3),dim=-2)
        # # attn_weights=torch.cat((attn_weights,attn_weights2),dim=1)

        # # shift back
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_states,key_states,value_states,topklist=[4],patchsize=[1024],w=1024,flash_use=True)
        attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[512],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_l,key_l,value_l,topklist=[16],patchsize=[64],w=64,mergep=4,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_l,key_l,value_l,topklist=[12],patchsize=[128],w=128,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,flash_use=True)
        # attn_outputs,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[128],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dtt(self,query_r,key_r,value_r,topklist=[2],patchsize=[1024],w=1024,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[128],patchsize=[8],w=8,mergep=16,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_states,key_states,value_states,topklist=[64*4],patchsize=[16],w=16,mergep=64,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttms(self,query_states,key_states,value_states,topklist=[8*4,4*4,4*4],patchsize=[16,64,256],w=16,mergep=16,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttms(self,query_states,key_states,value_states,topklist=[8,8,4],patchsize=[8,32,256],w=8,mergep=2,flash_use=True)
        # attn_output,attn_weights=forward_mss_local_inline_for_cccp_dttm(self,query_r,key_r,value_r,topklist=[4],patchsize=[1024],w=1024,mergep=1,flash_use=True)
        # attn_output=torch.cat((attn_outputs,attn_output),dim=-2)
        # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # if needed_pad:
        #     attn_output=attn_output[:,:orig_seq_len]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


