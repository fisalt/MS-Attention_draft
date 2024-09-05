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
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
flash_attn_version="2.1.0"
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_func
)
# import logger
import math
from typing import List, Optional, Tuple, Union

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # is_flash_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, LlamaDecoderLayer, LlamaRMSNorm
# LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast,BaseModelOutputWithPast
from flash_attn.bert_padding import unpad_input, pad_input
import math
from transformers.models.llama.configuration_llama import LlamaConfig
from local_attention import LocalAttention
from deepspeed import DeepSpeedEngine


class Fit_Func(nn.Module):
    def __init__(self,input_dim=4096,output_dim=4096,ratio=2):
        super(Fit_Func, self).__init__()
        self.fit=nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim*ratio),
            nn.GELU(),
            nn.Linear(input_dim*ratio, output_dim),
        )
        # self.linear = nn.Linear(dim, dim*ratio)

    def forward(self, x):
        return self.fit(x)

class SharedFit_Func(nn.Module):
    def __init__(self, dim=4096, ratio=2, bias=True):
        super(SharedFit_Func, self).__init__()
        self.norm=nn.LayerNorm(dim)
        self.in_features = dim
        self.out_features = dim*ratio
        self.weightu = nn.Parameter(torch.Tensor(dim*ratio, dim))
        self.weightd = nn.Parameter(torch.Tensor(dim, dim*ratio))
        if bias:
            self.biasu = nn.Parameter(torch.Tensor(dim*ratio))
            self.biasd = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weightd, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weightu, a=math.sqrt(5))
        if self.biasd is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weightd)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.biasd, -bound, bound)
        if self.biasu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weightu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.biasu, -bound, bound)

    def forward(self, input):
        input=nn.functional.linear(input, self.weightu, self.biasu)
        input=nn.functional.gelu(input)
        input=nn.functional.linear(input, self.weightd, self.biasd)
        return input

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config,chunksize=4096,max_len=32768):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.pos_head = nn.Linear(config.hidden_size, 1, bias=False)
        # self.pos_norm=nn.LayerNorm(config.hidden_size)
        # self.pos_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, 100, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(100, 1, bias=False)
        # )

        # Initialize weights and apply final processing
        self.max_len=65536
        self.max_len=32768
        self.max_len=262144
        self.max_len=1048576
        self.max_len=524288
        self.max_len=131072
        self.max_len=524288
        self.max_len=max_len
        self.chunksize=4096
        self.chunksize=16384
        self.chunksize=8192
        self.chunksize=chunksize
        self.patchsize=[1,2,8]
        self.mss_wdow=2048
        self.post_init()
        print(self.chunksize)
        print(self.max_len)
        self.trans=nn.ModuleList()
        # self.fit_func=Fit_Func(2*(config.hidden_size//config.num_attention_heads),2*(config.hidden_size//config.num_attention_heads),ratio=4)
        for i in range(((self.max_len-1)//self.chunksize+1)*2):
            self.trans.append(Fit_Func((config.hidden_size//config.num_attention_heads),(config.hidden_size//config.num_attention_heads),ratio=4))
        # self.fit_func=Fit_Func(config.hidden_size)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        chunksize=self.chunksize
        # self.config.output_hidden_states=True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states=[]
        outputss=BaseModelOutputWithPast(
                    last_hidden_state=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                    past_key_values=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                    hidden_states=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                    attentions=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                    )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print(input_ids.shape)
        use_cache = True
        past_key_values=[]
        # past_key_values=None
        logitss=[]
        loss = 0
        losslist = []
        # for i in range((self.max_len-1)//chunksize+1):
        for i in range((input_ids.shape[-1]-1)//chunksize+1):

            if i==0:
                outputs = self.model(
                    # input_ids=input_ids[...,i*chunksize:(i+1)*chunksize].cuda().to(torch.bfloat16),
                    input_ids=input_ids[...,i*chunksize:(i+1)*chunksize].cuda(),
                    attention_mask=attention_mask,
                    # position_ids=position_ids,
                    position_ids=0,
                    past_key_values=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # past_key_values=outputs.past_key_values
                # past_key_values=outputs[1]
            else:
                min_indx=min(self.patchsize[-1]*self.mss_wdow,i*chunksize)
                # print(input_ids[...,i*chunksize:(i+1)*chunksize].shape)
                # if self.patchsize[-1]*self.mss_wdow:
                outputs = self.model(
                    input_ids=input_ids[...,i*chunksize:(i+1)*chunksize].cuda(),
                    # input_ids=input_ids[...,i*chunksize-min_indx:(i+1)*chunksize],
                    attention_mask=attention_mask,
                    position_ids=i*chunksize,
                    # position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            past_key_values=outputs[1]
            # print(outputs[1])
            # print(outputs[1][0][0].shape)
            # print(past_key_values[0][:128])
            hidden_state=outputs[0]
            # hidden_states.append(hidden_state)
            if return_dict is not None:
                pass
                outputss=BaseModelOutputWithPast(
                    last_hidden_state=torch.cat((outputss.last_hidden_state,hidden_state),dim=1),
                    past_key_values=outputs.past_key_values if outputs.past_key_values is not None else None,
                    # past_key_values=torch.cat((outputss.past_key_values,outputs.past_key_values),dim=1) if outputs.past_key_values is not None else None,
                    hidden_states=outputs.hidden_states if outputs.hidden_states is not None else None,
                    # hidden_states=torch.cat((outputss.hidden_states,outputs.hidden_states),dim=1) if outputs.hidden_states is not None else None,
                    attentions=torch.cat((outputss.attentions,outputs.attentions),dim=1) if outputs.attentions is not None else None,
                )
            # torch.cuda.empty_cache()


            # hidden_states = torch.cat(hidden_states,dim=1)
            # outputs[1]=[]
            # past_key_values=[]
            # logitss=[]
            # hidden_states = outputs[0]
            # mse=nn.CrossEntropyLoss()

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                # logits = self.lm_head(hidden_states)
                logits = self.lm_head(hidden_state)
            logits = logits.float()
            # logitss.append(logits)
            # print(logits.shape)
            torch.cuda.empty_cache()
            # print("logits")
            # logits = torch.cat(logitss, dim=-2)
            # print(logits.shape)

            # loss = None
            if labels is not None:
                # print("loss")
                # Shift so that tokens < n predict n
                if i == (self.max_len-1)//chunksize:
                    shift_logits = logits[..., :chunksize-1, :].contiguous()
                    shift_labels = labels[..., i*chunksize+1:].contiguous().cuda()
                    # shift_labels = labels[..., i*chunksize+1:].contiguous()
                else:
                    shift_logits = logits[..., :chunksize, :].contiguous()
                    shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous().cuda()
                    # shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss1 = loss_fct(shift_logits, shift_labels)
                # loss1 = loss_fct(shift_logits, shift_labels)*shift_logits.shape[-2]/(input_ids.shape[-1])
                print(loss1)
                loss1 = loss1*shift_logits.shape[-2]/(input_ids.shape[-1])
            # DeepSpeedEngine.backward(loss1)
            # loss1.backward()
                losslist.append(loss1)
                loss += loss1
            if self.training:
                logitss.append(logits.cpu())
            else:
                logitss.append(logits)
            # logitss.append(logits)
            # logitss.append(logits)
        if labels is not None:
            # loss_total=loss/((self.max_len-1)//chunksize+1)
            loss_total=loss
            if self.training:
                losstt=losslist
            else:
                losstt=loss_total
        else:
            losstt=None
        logits = torch.cat(logitss, dim=-2)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (losstt,) + output if loss is not None else output
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=losstt,
            # loss=loss,
            logits=logits,
            past_key_values=None,
            # past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config,i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            # position_ids = position_ids.view(-1, seq_length).long()
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                position_ids, seq_length + position_ids, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        #     )
        #     padding_mask = None
        # else:
        #     if 0 in attention_mask:
        #         padding_mask = attention_mask
        #     else:
        #         padding_mask = None

        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )
        attention_mask=None
        padding_mask=None

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = use_cache

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask,use_cache=use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                # print(use_cache)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            # print(layer_outputs)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


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

from transformers import Trainer
from transformers.utils import is_apex_available,is_sagemaker_mp_enabled
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

class CustomTrainer(Trainer):
    def __init__(self, *args, custom_attribute=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_attribute = custom_attribute
        self.do_grad_scaling=True

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with super().compute_loss_context_manager():
            losslist = super().compute_loss(model, inputs)

        loss_total=0.0
        print(losslist)
        for loss in losslist:
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # if self.do_grad_scaling:
            #     self.scaler.scale(loss).backward()
            # elif self.use_apex:
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            loss_total+=loss
            torch.cuda.empty_cache()
        loss_total=loss_total

        return loss_total.detach() / self.args.gradient_accumulation_steps


class CustomTrainer2(Trainer):
    def __init__(self, *args, max_position_embeddings_Trainer=None, head_dim_Trainer=None, scaling_factor_Trainer=None, rope_theta_Trainer=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.custom_attribute = custom_attribute
        self.do_grad_scaling=True
        self.steps_sub=0
        # self.max_position_embeddings_Trainer=max_position_embeddings_Trainer
        self.max_position_embeddings_Trainer=max_position_embeddings_Trainer
        self.head_dim_Trainer=head_dim_Trainer
        self.scaling_factor_Trainer=scaling_factor_Trainer
        self.rope_theta_Trainer=rope_theta_Trainer

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with super().compute_loss_context_manager():
            loss = super().compute_loss(model, inputs)


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        # get_accelerator().empty_cache()
        # self.steps_sub+=1
        # if self.steps_sub%1000==0:
        #     # pass
        #     self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
        #             self.max_position_embeddings_Trainer,
        #             max_position_embeddings=self.head_dim_Trainer,
        #             scaling_factor=self.scaling_factor_Trainer,
        #             base=self.rope_theta_Trainer,
        #         )

        return loss.detach() / self.args.gradient_accumulation_steps
    



class CustomTrainer3(Trainer):
    def __init__(self, *args, steps_pos_t=500, max_position_embeddings_Trainer=None, head_dim_Trainer=None, scaling_factor_Trainer=64, rope_theta_Trainer=None, dynamic=1, **kwargs):
        super().__init__(*args, **kwargs)
        # self.custom_attribute = custom_attribute
        self.do_grad_scaling=True
        self.steps_pos=0
        self.steps_pos_t=steps_pos_t
        # self.max_position_embeddings_Trainer=max_position_embeddings_Trainer
        self.max_position_embeddings_Trainer=max_position_embeddings_Trainer
        self.head_dim_Trainer=head_dim_Trainer
        self.scaling_factor_Trainer=scaling_factor_Trainer
        self.dynamic=dynamic
        self.rope_theta_Trainer=rope_theta_Trainer

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with super().compute_loss_context_manager():
            loss = super().compute_loss(model, inputs)


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        torch.cuda.empty_cache()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        torch.cuda.empty_cache()
        # get_accelerator().empty_cache()
        self.steps_pos+=1
        if self.steps_pos%self.steps_pos_t==0:
            self.scaling_factor_Trainer=self.scaling_factor_Trainer*2
            # pass
            if self.dynamic==0:
                for i in range(len(model.module.model.model.layers)):
                    # model.model.layers[i].self_attn=LlamaAttention_mss_ccc_sps(config,seg=training_args.seg,merge=training_args.merge,top=training_args.top)
                    model.module.model.model.layers[i].self_attn.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                        self.head_dim_Trainer,
                        max_position_embeddings=self.max_position_embeddings_Trainer,
                        scaling_factor=self.scaling_factor_Trainer,
                        base=self.rope_theta_Trainer,
                        ).to(model.module.device)
            else:
                for i in range(len(model.module.model.model.layers)):
                    # model.model.layers[i].self_attn=LlamaAttention_mss_ccc_sps(config,seg=training_args.seg,merge=training_args.merge,top=training_args.top)
                    model.module.model.model.layers[i].self_attn.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                        self.head_dim_Trainer,
                        max_position_embeddings=self.max_position_embeddings_Trainer,
                        scaling_factor=self.scaling_factor_Trainer,
                        base=self.rope_theta_Trainer,
                        ).to(model.module.device)

        return loss.detach() / self.args.gradient_accumulation_steps
    





class CustomTrainer2_t(Trainer):
    def __init__(self, *args, max_position_embeddings_Trainer=None, head_dim_Trainer=None, scaling_factor_Trainer=None, rope_theta_Trainer=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.custom_attribute = custom_attribute
        self.do_grad_scaling=True
        self.steps_sub=0
        # self.max_position_embeddings_Trainer=max_position_embeddings_Trainer
        self.max_position_embeddings_Trainer=max_position_embeddings_Trainer
        self.head_dim_Trainer=head_dim_Trainer
        self.scaling_factor_Trainer=scaling_factor_Trainer
        self.rope_theta_Trainer=rope_theta_Trainer

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with super().compute_loss_context_manager():
            loss = super().compute_loss(model, inputs)


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        torch.cuda.empty_cache()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        torch.cuda.empty_cache()
        # get_accelerator().empty_cache()
        # self.steps_sub+=1
        # if self.steps_sub%1000==0:
        #     # pass
        #     self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
        #             self.max_position_embeddings_Trainer,
        #             max_position_embeddings=self.head_dim_Trainer,
        #             scaling_factor=self.scaling_factor_Trainer,
        #             base=self.rope_theta_Trainer,
        #         )

        return loss.detach() / self.args.gradient_accumulation_steps

def prepare_position_ids(device,seq_length):
    position_ids = torch.arange(
        0, seq_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids

def prepare_attnmask(module,inputs_embeds,seq_length,device,batch_size=1):
    attention_mask = torch.ones(
            (batch_size, seq_length), dtype=torch.bool, device=device
        )
    padding_mask = None
    # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = _prepare_decoder_attention_mask(
        module, attention_mask, (batch_size, seq_length), inputs_embeds, 0
    )
    return attention_mask


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_LMt(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states=[]
    outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                past_key_values=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                hidden_states=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                attentions=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # print(input_ids.shape)
    # Define CUDA streams for asynchronous operations
    stream_gpu1 = torch.cuda.Stream(device=0)
    stream_gpu2 = torch.cuda.Stream(device=1)

    # Split model into segments (example)
    # model_segments = [model.layer[i] for i in range(len(model.layer))]

    outputs = []
    with torch.no_grad():
        with torch.cuda.stream(stream_gpu1):
            # model_segments2 = model.model.layer[i].to(1)
            embed_tokens=self.model.embed_tokens.to(0)
        output_gpu1=embed_tokens(input_ids.to(0))

        seq_length=input_ids.shape[-1]
        position_ids=prepare_position_ids(output_gpu1.device,seq_length)
        # attention_mask=prepare_attnmask(model.model,input_ids,seq_length,input_ids.device,input_ids.shape[0])
        del input_ids
        del embed_tokens
        torch.cuda.empty_cache()

        for i in range(0, 32, 2):
            
            with torch.cuda.stream(stream_gpu2):
                model_segments2 = self.model.layers[i].to(1)
            torch.cuda.synchronize(2)
            output_gpu1 = output_gpu1.to(1)
            # attention_mask = attention_mask.to(1)
            # position_ids = position_ids.to(1)
            # torch.cuda.empty_cache()
            # Load and compute on GPU 1
            # position_ids=prepare_position_ids(output_gpu1.device,seq_length)
            position_ids=prepare_position_ids(output_gpu1.device,seq_length)
            
            output_gpu2 = model_segments2(output_gpu1,attention_mask=None,position_ids=position_ids)[0]
            # 清理 GPU 0 上的显存
            del model_segments2  # 删除不再需要的模型部分
            del output_gpu1      # 删除不再需要的张量
            del position_ids      # 删除不再需要的张量
            torch.cuda.empty_cache()

            with torch.cuda.stream(stream_gpu1):
                    model_segments1 = self.model.layers[i+1].to(2)

            torch.cuda.synchronize(1)
            output_gpu2 = output_gpu2.to(2)
            # attention_mask = attention_mask.to(0)
            # position_ids = position_ids.to(0)
            position_ids=prepare_position_ids(output_gpu2.device,seq_length)

            output_gpu1 = model_segments1(output_gpu2,attention_mask=None,position_ids=position_ids)[0]

            # 清理 GPU 1 上的显存
            del model_segments1  # 删除不再需要的模型部分
            del output_gpu2      # 删除不再需要的张量
            del position_ids      # 删除不再需要的张量
            torch.cuda.empty_cache()
            
        with torch.cuda.stream(stream_gpu2):
            # model_segments2 = model.model.layer[i].to(1)
            norm=self.model.norm.to(2)
            lm_head=self.lm_head.to(2)
        hidden_state=norm(output_gpu1)  
        logits=lm_head(hidden_state).to(0)
        hidden_state=hidden_state.to(0)
        del norm
        del lm_head
        torch.cuda.empty_cache()
        past_key_values=None
        outputs=[None,None]
        # print(outputs[1])
        # print(outputs[1][0][0].shape)
        # print(past_key_values[0][:128])
        # hidden_state=outputs[0]
        # hidden_states.append(hidden_state)
        if return_dict is not None:
            pass
            outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.cat((outputss.last_hidden_state.cpu(),hidden_state.cpu()),dim=1),
                past_key_values=None,
                # past_key_values=torch.cat((outputss.past_key_values,outputs.past_key_values),dim=1) if outputs.past_key_values is not None else None,
                hidden_states=None,
                # hidden_states=torch.cat((outputss.hidden_states,outputs.hidden_states),dim=1) if outputs.hidden_states is not None else None,
                attentions=None,
            )
        # torch.cuda.empty_cache()


        # hidden_states = torch.cat(hidden_states,dim=1)
        # outputs[1]=[]
        # past_key_values=[]
        # logitss=[]
        # hidden_states = outputs[0]
        # mse=nn.CrossEntropyLoss()

        # if self.config.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        #     # logits = self.lm_head(hidden_states)
        #     logits = self.lm_head(hidden_state)
        # logits = logits.float()
        # logitss.append(logits)
        # print(logits.shape)
        torch.cuda.empty_cache()
        # print("logits")
        # logits = torch.cat(logitss, dim=-2)
        # print(logits.shape)

        # loss = None
        losslist=[]
        if labels is not None:
            chunksize=4096
            # print("loss")
            # Shift so that tokens < n predict n
            if i == (self.max_len-1)//chunksize:
                shift_logits = logits[..., :chunksize-1, :].contiguous()
                shift_labels = labels[..., i*chunksize+1:].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:].contiguous()
            else:
                shift_logits = logits[..., :chunksize, :].contiguous()
                shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss1 = loss_fct(shift_logits, shift_labels)
            # loss1 = loss_fct(shift_logits, shift_labels)*shift_logits.shape[-2]/(input_ids.shape[-1])
            print(loss1)
            loss1 = loss1*shift_logits.shape[-2]/(input_ids.shape[-1])
        # DeepSpeedEngine.backward(loss1)
        # loss1.backward()
            losslist.append(loss1)
            loss += loss1
        # logitss.append(logits)
        # logitss.append(logits)
    if labels is not None:
        # loss_total=loss/((self.max_len-1)//chunksize+1)
        loss_total=loss
        if self.training:
            losstt=losslist
        else:
            losstt=loss_total
    else:
        losstt=None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (losstt,) + output if loss is not None else output
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=losstt,
        # loss=loss,
        logits=logits,
        past_key_values=None,
        # past_key_values=outputs.past_key_values,
        hidden_states=None,
        attentions=None,
    )



def process_linear_layer_in_chunks(layer, inputs, chunk_size):
    outputs = []
    for i in range(0, inputs.shape[1], chunk_size):
        chunk = inputs[:, i:i+chunk_size]
        chunk_output = layer(chunk.to('cuda'))
        outputs.append(chunk_output.cpu())
        # del chunk_output
    return torch.cat(outputs, dim=1).cpu()

def process_attention_layer_in_chunks(attention_layer, queries, keys, values, chunk_size):
    num_heads = queries.size(2)  # Assuming the queries have shape [batch_size, seq_length, num_heads, head_dim]
    head_dim = queries.size(-1)
    
    # attn_output_his = torch.zeros_like(queries).cpu()  # Initialize history to accumulate attention outputs
    # Slsek = torch.full((queries.size(0), num_heads), float('-inf')).cpu()  # Initialize Slsek to negative infinity
    attn_output_his = None  # Initialize history to accumulate attention outputs
    Slsek = None  # Initialize Slsek to negative infinity
    
    for i in range(0, queries.size(1), chunk_size):
        q_chunk = queries[:, i:i+chunk_size].to('cuda')
        chunk_outputs = []
        attn_output_his = None  # Initialize history to accumulate attention outputs
        Slsek = None  # Initialize Slsek to negative infinity
        
        for j in range(0, keys.size(1), chunk_size):
            k_chunk = keys[:, j:j+chunk_size].to('cuda')
            v_chunk = values[:, j:j+chunk_size].to('cuda')
            
            # Perform attention computation
            # attn_output, attn_scores = attention_layer(q_chunk, k_chunk, v_chunk, need_weights=True)
            attn_output, slek, _ = flash_attn_func(q_chunk, k_chunk, v_chunk, return_attn_probs=True)
            
            # Move the result to CPU to accumulate
            # attn_output = attn_output
            # slek = attn_scores.logsumexp(dim=-1)
            
            # LogSumExp trick to update Slsek and attn_output_his
            if Slsek is None:
                Slsek = slek
                attn_output_his = attn_output
            else:
                lsesk = torch.logsumexp(torch.cat((slek[..., None], Slsek[..., None]), dim=-1), dim=-1).detach()
                attn_output_his = attn_output_his * (1 / (torch.exp(lsesk - Slsek)))[..., None] + attn_output * (1 / (torch.exp(lsesk - slek)))[..., None]
                Slsek = lsesk
        
        chunk_outputs.append(attn_output_his)
    
    return torch.cat(chunk_outputs, dim=1)


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_LMc(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states=[]
    outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                past_key_values=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                hidden_states=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                attentions=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # print(input_ids.shape)
    # Define CUDA streams for asynchronous operations

    # Split model into segments (example)
    # model_segments = [model.layer[i] for i in range(len(model.layer))]

    outputs = []
    cuda_device=0
    chunk_size=256*1024
    with torch.no_grad():
        # with torch.cuda.stream(stream_gpu1):
        #     # model_segments2 = model.model.layer[i].to(1)
        #     embed_tokens=model.model.embed_tokens.to(cuda_device)
        embed_tokens=self.model.embed_tokens.to(cuda_device)
        outputs=process_linear_layer_in_chunks(embed_tokens,input_ids,chunk_size//4)
        # print(outputs.shape)
            # outputs.append(output_gpu1.to(1))

        # output_gpu1 = torch.cat(outputs,dim=-2).to(1)
        # output_gpu1 = torch.cat(outputs,dim=-2)
        # seq_length=input_ids.shape[-1]
        # position_ids=prepare_position_ids(outputs.device,input_sequence.shape[-2])
        position_ids=prepare_position_ids(outputs.device,input_ids.shape[-1])
        # attention_mask=prepare_attnmask(model.model,input_ids,seq_length,input_ids.device,input_ids.shape[0])
        del input_ids
        del embed_tokens
        # outputs=[]
        torch.cuda.empty_cache()
        res=outputs

        for i in range(0, 32, 1):
            
            # with torch.cuda.stream(stream_gpu2):
            #     model_norm = model.model.layers[i].input_layernorm.to(cuda_device)
            #     model_attn = model.model.layers[i].self_attn.to(cuda_device)
            #     # model_segments2 = model.model.layers[i].to(1)
            # torch.cuda.synchronize(0)
            # output_gpu1 = output_gpu1.to(1)
            model_norm = self.model.layers[i].input_layernorm.to(cuda_device)
            outputs1=process_linear_layer_in_chunks(model_norm,outputs,chunk_size)


            # position_ids=prepare_position_ids(outputs.device,input_sequence.shape[-2])
            model_attn = self.model.layers[i].self_attn.to(cuda_device)
            # outputs1=model_attn(outputs1.to(cuda_device),attention_mask=None,position_ids=position_ids.to(cuda_device))[0].cpu()
            outputs1=model_attn(outputs1,attention_mask=None,position_ids=position_ids)[0]
            # output_gpu1 = output_gpu1.to(1)
            # attention_mask = attention_mask.to(1)
            # position_ids = position_ids.to(1)
            # torch.cuda.empty_cache()
            # Load and compute on GPU 1
            # position_ids=prepare_position_ids(output_gpu1.device,seq_length)

            # position_ids=prepare_position_ids(output_gpu1.device,seq_length)
            
            # output_gpu2 = model_norm(output_gpu1)
            # for j in range((seq_length-1)//(seg)+1):
            #     for k in range((seq_length-1)//(seg)+1):
            #         output_gputemp = model_attn(output_gpu2,attention_mask=None,position_ids=position_ids)[0]
            # # output_gpu2 = model_attn(output_gpu2,attention_mask=None,position_ids=position_ids)[0]
            # output_gpu2 = output_gpu1+output_gpu2

            outputs = outputs1+res
            # 清理 GPU 0 上的显存
            del model_norm  # 删除不再需要的模型部分
            del model_attn  # 删除不再需要的模型部分
            # del output_gpu1      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            torch.cuda.empty_cache()

            # with torch.cuda.stream(stream_gpu1):
            #         # model_segments1 = model.model.layers[i+1].to(0)
            #         model_mlp = model.model.layers[i].mlp.to(0)
            #         model_postnorm = model.model.layers[i].post_attention_layernorm.to(0)

            # torch.cuda.synchronize(1)
            # output_gpu2 = output_gpu2.to(0)
            # # attention_mask = attention_mask.to(0)
            # # position_ids = position_ids.to(0)
            # # position_ids=prepare_position_ids(output_gpu2.device,seq_length)

            model_postnorm = self.model.layers[i].post_attention_layernorm.to(cuda_device)
            outputsn=process_linear_layer_in_chunks(model_postnorm,outputs,chunk_size)

            model_mlp = self.model.layers[i].mlp.to(cuda_device)
            outputsn=process_linear_layer_in_chunks(model_mlp,outputsn,chunk_size//2)
            
            outputs=res+outputsn
            res=outputs

            # model_postnorm = self.model.layers[i].post_attention_layernorm.to(cuda_device)
            # outputs=process_linear_layer_in_chunks(model_postnorm,outputs,chunk_size)


            # outputs=[]
            # output_gpu1 = model_postnorm(output_gpu2)
            # for j in range((seq_length-1)//(seg)+1):
            #     output_gputemp=model_mlp(output_gpu1[...,j*(seg):(j+1)*(seg),:])
            #     # outputs.append(output_gputemp.cpu())
            #     outputs.append(output_gputemp+output_gpu2[...,j*(seg):(j+1)*(seg),:])
            #     del output_gputemp
            #     torch.cuda.empty_cache()
            #     print(j)
            # # output_gpu1 = model_mlp(output_gpu1)
            # del output_gpu1      # 删除不再需要的张量
            # output_gpu1 = torch.cat(outputs,dim=-2)
            # output_gpu1 = output_gpu1+output_gpu2

            # 清理 GPU 1 上的显存
            del model_postnorm  # 删除不再需要的模型部分
            del model_mlp  # 删除不再需要的模型部分
            # del output_gpu2      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            # outputs=[]
            torch.cuda.empty_cache()
            
        # with torch.cuda.stream(stream_gpu2):
        #     # model_segments2 = model.model.layer[i].to(1)
        #     norm=model.model.norm.to(0)
        #     lm_head=model.lm_head.to(0)
        # outputs=[]
        # output_gpu1=norm(output_gpu1)  
        # for i in range((seq_length-1)//seg+1):
        #     output_gpu2=lm_head(output_gpu1[i*seg:(i+1)*seg].to(0))
        #     # outputs.append(output_gpu1.cpu())
        #     outputs.append(output_gpu2.cpu())
        # # output_gpu1=lm_head(output_gpu1)  

        norm=self.model.norm.to(cuda_device)
        hidden_state=process_linear_layer_in_chunks(norm,outputs,chunk_size)
        lm_head=self.lm_head.to(cuda_device)
        logits=process_linear_layer_in_chunks(lm_head,hidden_state,chunk_size//4)

        del norm
        del lm_head
        torch.cuda.empty_cache()
            
        # hidden_state=norm(output_gpu1)  
        # logits=lm_head(hidden_state).to(0)
        # hidden_state=hidden_state.to(0)
        # del norm
        # del lm_head
        # torch.cuda.empty_cache()
        past_key_values=None
        outputs=[None,None]
        # print(outputs[1])
        # print(outputs[1][0][0].shape)
        # print(past_key_values[0][:128])
        # hidden_state=outputs[0]
        # hidden_states.append(hidden_state)
        if return_dict is not None:
            pass
            outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.cat((outputss.last_hidden_state.cpu(),hidden_state.cpu()),dim=1),
                past_key_values=None,
                # past_key_values=torch.cat((outputss.past_key_values,outputs.past_key_values),dim=1) if outputs.past_key_values is not None else None,
                hidden_states=None,
                # hidden_states=torch.cat((outputss.hidden_states,outputs.hidden_states),dim=1) if outputs.hidden_states is not None else None,
                attentions=None,
            )
        # torch.cuda.empty_cache()


        # hidden_states = torch.cat(hidden_states,dim=1)
        # outputs[1]=[]
        # past_key_values=[]
        # logitss=[]
        # hidden_states = outputs[0]
        # mse=nn.CrossEntropyLoss()

        # if self.config.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        #     # logits = self.lm_head(hidden_states)
        #     logits = self.lm_head(hidden_state)
        # logits = logits.float()
        # logitss.append(logits)
        # print(logits.shape)
        torch.cuda.empty_cache()
        # print("logits")
        # logits = torch.cat(logitss, dim=-2)
        # print(logits.shape)

        # loss = None
        losslist=[]
        if labels is not None:
            chunksize=4096
            # print("loss")
            # Shift so that tokens < n predict n
            if i == (self.max_len-1)//chunksize:
                shift_logits = logits[..., :chunksize-1, :].contiguous().cuda()
                shift_labels = labels[..., i*chunksize+1:].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:].contiguous()
            else:
                shift_logits = logits[..., :chunksize, :].contiguous().cuda()
                shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss1 = loss_fct(shift_logits, shift_labels)
            # loss1 = loss_fct(shift_logits, shift_labels)*shift_logits.shape[-2]/(input_ids.shape[-1])
            print(loss1)
            loss1 = loss1*shift_logits.shape[-2]/(input_ids.shape[-1])
        # DeepSpeedEngine.backward(loss1)
        # loss1.backward()
            losslist.append(loss1)
            loss += loss1
        # logitss.append(logits)
        # logitss.append(logits)
    if labels is not None:
        # loss_total=loss/((self.max_len-1)//chunksize+1)
        loss_total=loss
        if self.training:
            losstt=losslist
        else:
            losstt=loss_total
    else:
        losstt=None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (losstt,) + output if loss is not None else output
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=losstt,
        # loss=loss,
        logits=logits,
        past_key_values=None,
        # past_key_values=outputs.past_key_values,
        hidden_states=None,
        attentions=None,
    )


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_LMca(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states=[]
    outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                past_key_values=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                hidden_states=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                attentions=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # print(input_ids.shape)
    # Define CUDA streams for asynchronous operations

    # Split model into segments (example)
    # model_segments = [model.layer[i] for i in range(len(model.layer))]

    outputs = []
    cuda_device=0
    chunk_size=256*1024
    with torch.no_grad():
        # with torch.cuda.stream(stream_gpu1):
        #     # model_segments2 = model.model.layer[i].to(1)
        #     embed_tokens=model.model.embed_tokens.to(cuda_device)
        embed_tokens=self.model.embed_tokens.to(cuda_device)
        outputs=process_linear_layer_in_chunks(embed_tokens,input_ids,chunk_size//4)
        # print(outputs.shape)
            # outputs.append(output_gpu1.to(1))

        # output_gpu1 = torch.cat(outputs,dim=-2).to(1)
        # output_gpu1 = torch.cat(outputs,dim=-2)
        # seq_length=input_ids.shape[-1]
        # position_ids=prepare_position_ids(outputs.device,input_sequence.shape[-2])
        position_ids=prepare_position_ids(outputs.device,input_ids.shape[-1])
        # attention_mask=prepare_attnmask(model.model,input_ids,seq_length,input_ids.device,input_ids.shape[0])
        del input_ids
        del embed_tokens
        # outputs=[]
        torch.cuda.empty_cache()
        res=outputs

        for i in range(0, 32, 1):

            # model_layers=self.model.layers[i].to(cuda_device)
            # outputs=model_layers(outputs.to(cuda_device),attention_mask=None,position_ids=position_ids.to(cuda_device))[0]
            
            model_norm = self.model.layers[i].input_layernorm.to(cuda_device)
            outputs1=process_linear_layer_in_chunks(model_norm,outputs,chunk_size)


            # # position_ids=prepare_position_ids(outputs.device,input_sequence.shape[-2])
            model_attn = self.model.layers[i].self_attn.to(cuda_device)
            # outputs1=model_attn(outputs1.to(cuda_device),attention_mask=None,position_ids=position_ids.to(cuda_device))[0].cpu()
            outputs1=model_attn(outputs1,attention_mask=None,position_ids=position_ids)[0]

            outputs = outputs1+res
            # 清理 GPU 0 上的显存
            del model_norm  # 删除不再需要的模型部分
            del model_attn  # 删除不再需要的模型部分
            # del output_gpu1      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            torch.cuda.empty_cache()
            res=outputs

            model_postnorm = self.model.layers[i].post_attention_layernorm.to(cuda_device)
            outputsn=process_linear_layer_in_chunks(model_postnorm,outputs,chunk_size)

            model_mlp = self.model.layers[i].mlp.to(cuda_device)
            outputsn=process_linear_layer_in_chunks(model_mlp,outputsn,chunk_size//2)
            
            outputs=res+outputsn
            res=outputs

            # 清理 GPU 1 上的显存
            del model_postnorm  # 删除不再需要的模型部分
            del model_mlp  # 删除不再需要的模型部分
            # del output_gpu2      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            # outputs=[]
            # del model_layers
            torch.cuda.empty_cache()

        norm=self.model.norm.to(cuda_device)
        hidden_state=process_linear_layer_in_chunks(norm,outputs,chunk_size)
        lm_head=self.lm_head.to(cuda_device)
        logits=process_linear_layer_in_chunks(lm_head,hidden_state,chunk_size//4)

        del norm
        del lm_head
        torch.cuda.empty_cache()
            
        past_key_values=None
        outputs=[None,None]

        if return_dict is not None:
            pass
            outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.cat((outputss.last_hidden_state.cpu(),hidden_state.cpu()),dim=1),
                past_key_values=None,
                # past_key_values=torch.cat((outputss.past_key_values,outputs.past_key_values),dim=1) if outputs.past_key_values is not None else None,
                hidden_states=None,
                # hidden_states=torch.cat((outputss.hidden_states,outputs.hidden_states),dim=1) if outputs.hidden_states is not None else None,
                attentions=None,
            )
        # torch.cuda.empty_cache()


        torch.cuda.empty_cache()

        # loss = None
        losslist=[]
        if labels is not None:
            chunksize=4096
            # print("loss")
            # Shift so that tokens < n predict n
            if i == (self.max_len-1)//chunksize:
                shift_logits = logits[..., :chunksize-1, :].contiguous().cuda()
                shift_labels = labels[..., i*chunksize+1:].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:].contiguous()
            else:
                shift_logits = logits[..., :chunksize, :].contiguous().cuda()
                shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss1 = loss_fct(shift_logits, shift_labels)
            # loss1 = loss_fct(shift_logits, shift_labels)*shift_logits.shape[-2]/(input_ids.shape[-1])
            print(loss1)
            loss1 = loss1*shift_logits.shape[-2]/(input_ids.shape[-1])
        # DeepSpeedEngine.backward(loss1)
        # loss1.backward()
            losslist.append(loss1)
            loss += loss1
        # logitss.append(logits)
        # logitss.append(logits)
    if labels is not None:
        # loss_total=loss/((self.max_len-1)//chunksize+1)
        loss_total=loss
        if self.training:
            losstt=losslist
        else:
            losstt=loss_total
    else:
        losstt=None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (losstt,) + output if loss is not None else output
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=losstt,
        # loss=loss,
        logits=logits,
        past_key_values=None,
        # past_key_values=outputs.past_key_values,
        hidden_states=None,
        attentions=None,
    )



@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_LMca(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states=[]
    outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                past_key_values=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                hidden_states=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                attentions=torch.tensor([],device=input_ids.device,dtype=input_ids.dtype),
                )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # print(input_ids.shape)
    # Define CUDA streams for asynchronous operations

    # Split model into segments (example)
    # model_segments = [model.layer[i] for i in range(len(model.layer))]

    outputs = []
    cuda_device=0
    chunk_size=256*1024
    with torch.no_grad():
        # with torch.cuda.stream(stream_gpu1):
        #     # model_segments2 = model.model.layer[i].to(1)
        #     embed_tokens=model.model.embed_tokens.to(cuda_device)
        embed_tokens=self.model.embed_tokens.to(cuda_device)
        outputs=process_linear_layer_in_chunks(embed_tokens,input_ids,chunk_size//4)
        # print(outputs.shape)
            # outputs.append(output_gpu1.to(1))

        # output_gpu1 = torch.cat(outputs,dim=-2).to(1)
        # output_gpu1 = torch.cat(outputs,dim=-2)
        # seq_length=input_ids.shape[-1]
        # position_ids=prepare_position_ids(outputs.device,input_sequence.shape[-2])
        position_ids=prepare_position_ids(outputs.device,input_ids.shape[-1])
        # attention_mask=prepare_attnmask(model.model,input_ids,seq_length,input_ids.device,input_ids.shape[0])
        del input_ids
        del embed_tokens
        # outputs=[]
        torch.cuda.empty_cache()
        res=outputs

        for i in range(0, 32, 1):

            # model_layers=self.model.layers[i].to(cuda_device)
            # outputs=model_layers(outputs.to(cuda_device),attention_mask=None,position_ids=position_ids.to(cuda_device))[0]
            
            model_norm = self.model.layers[i].input_layernorm.to(cuda_device)
            outputs1=process_linear_layer_in_chunks(model_norm,outputs,chunk_size)


            # # position_ids=prepare_position_ids(outputs.device,input_sequence.shape[-2])
            model_attn = self.model.layers[i].self_attn.to(cuda_device)
            # outputs1=model_attn(outputs1.to(cuda_device),attention_mask=None,position_ids=position_ids.to(cuda_device))[0].cpu()
            outputs1=model_attn(outputs1,attention_mask=None,position_ids=position_ids)[0]

            outputs = outputs1+res
            # 清理 GPU 0 上的显存
            del model_norm  # 删除不再需要的模型部分
            del model_attn  # 删除不再需要的模型部分
            # del output_gpu1      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            torch.cuda.empty_cache()
            res=outputs

            model_postnorm = self.model.layers[i].post_attention_layernorm.to(cuda_device)
            outputsn=process_linear_layer_in_chunks(model_postnorm,outputs,chunk_size)

            model_mlp = self.model.layers[i].mlp.to(cuda_device)
            outputsn=process_linear_layer_in_chunks(model_mlp,outputsn,chunk_size//2)
            
            outputs=res+outputsn
            res=outputs

            # 清理 GPU 1 上的显存
            del model_postnorm  # 删除不再需要的模型部分
            del model_mlp  # 删除不再需要的模型部分
            # del output_gpu2      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            # del position_ids      # 删除不再需要的张量
            # outputs=[]
            # del model_layers
            torch.cuda.empty_cache()

        norm=self.model.norm.to(cuda_device)
        hidden_state=process_linear_layer_in_chunks(norm,outputs,chunk_size)
        lm_head=self.lm_head.to(cuda_device)
        logits=process_linear_layer_in_chunks(lm_head,hidden_state,chunk_size//4)

        del norm
        del lm_head
        torch.cuda.empty_cache()
            
        past_key_values=None
        outputs=[None,None]

        if return_dict is not None:
            pass
            outputss=BaseModelOutputWithPast(
                last_hidden_state=torch.cat((outputss.last_hidden_state.cpu(),hidden_state.cpu()),dim=1),
                past_key_values=None,
                # past_key_values=torch.cat((outputss.past_key_values,outputs.past_key_values),dim=1) if outputs.past_key_values is not None else None,
                hidden_states=None,
                # hidden_states=torch.cat((outputss.hidden_states,outputs.hidden_states),dim=1) if outputs.hidden_states is not None else None,
                attentions=None,
            )
        # torch.cuda.empty_cache()


        torch.cuda.empty_cache()

        # loss = None
        losslist=[]
        if labels is not None:
            chunksize=4096
            # print("loss")
            # Shift so that tokens < n predict n
            if i == (self.max_len-1)//chunksize:
                shift_logits = logits[..., :chunksize-1, :].contiguous().cuda()
                shift_labels = labels[..., i*chunksize+1:].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:].contiguous()
            else:
                shift_logits = logits[..., :chunksize, :].contiguous().cuda()
                shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous().cuda()
                # shift_labels = labels[..., i*chunksize+1:(i+1)*chunksize+1].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss1 = loss_fct(shift_logits, shift_labels)
            # loss1 = loss_fct(shift_logits, shift_labels)*shift_logits.shape[-2]/(input_ids.shape[-1])
            print(loss1)
            loss1 = loss1*shift_logits.shape[-2]/(input_ids.shape[-1])
        # DeepSpeedEngine.backward(loss1)
        # loss1.backward()
            losslist.append(loss1)
            loss += loss1
        # logitss.append(logits)
        # logitss.append(logits)
    if labels is not None:
        # loss_total=loss/((self.max_len-1)//chunksize+1)
        loss_total=loss
        if self.training:
            losstt=losslist
        else:
            losstt=loss_total
    else:
        losstt=None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (losstt,) + output if loss is not None else output
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=losstt,
        # loss=loss,
        logits=logits,
        past_key_values=None,
        # past_key_values=outputs.past_key_values,
        hidden_states=None,
        attentions=None,
    )

