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
    is_flash_attn_available,
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

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
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
        self.max_len=32768
        self.chunksize=4096
        self.chunksize=16384
        self.chunksize=8192
        self.chunksize=16384
        self.patchsize=[1,2,8]
        self.mss_wdow=2048
        self.post_init()

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
        print(input_ids.shape)
        use_cache = True
        past_key_values=[]
        # past_key_values=None
        logitss=[]
        loss = 0
        losslist = []
        for i in range((self.max_len-1)//chunksize+1):

            if i==0:
                outputs = self.model(
                    # input_ids=input_ids[...,i*chunksize:(i+1)*chunksize].cuda().to(torch.bfloat16),
                    input_ids=input_ids[...,i*chunksize:(i+1)*chunksize].cuda(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
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
                print(input_ids[...,i*chunksize:(i+1)*chunksize].shape)
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
            print(logits.shape)
            torch.cuda.empty_cache()
            print("logits")
            # logits = torch.cat(logitss, dim=-2)
            # print(logits.shape)

            # loss = None
            if labels is not None:
                print("loss")
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
                print(loss1)
            # DeepSpeedEngine.backward(loss1)
            # loss1.backward()
            losslist.append(loss1)
            loss += loss1
            logitss.append(logits.cpu())
            # logitss.append(logits)
        loss_total=loss/self.max_len*chunksize
        loss=loss1
        logits = torch.cat(logitss, dim=-2)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (losslist,) + output if loss is not None else output
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=losslist,
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
        loss_total=loss_total/len(losslist)

        return loss_total.detach() / self.args.gradient_accumulation_steps
