# Copyright (c) 2023, Tri Dao.

import math
import re

from collections import OrderedDict

import torch
import torch.nn.functional as F

from transformers import GPT2Config, GPTJConfig


def remap_state_dict_hf_gptj(state_dict, config):
    breakpoint()
    def key_mapping_model(key):
        key = re.sub(r'^model.decoder.', 'transformer.', key)
        return key
    state_dict = OrderedDict((key_mapping_model(k), v) for k, v in state_dict.items())
    # Word embedding and position embedding
    def key_mapping_emb(key):
        key = re.sub(r'^transformer.embed_tokens.', 'transformer.embeddings.word_embeddings.', key)
        # The OPT-350m model uses has project_in and project_out
        key = re.sub(r'^transformer.project_in.', 'transformer.embeddings.project_in.', key)
        key = re.sub(r'^transformer.project_out.', 'project_out.', key)
        key = re.sub(r'^transformer.embed_positions.',
                     'transformer.embeddings.position_embeddings.', key)
        return key
    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    # OPT uses the first 2 indices of pos_emb for padding tokens
    pos_embeddings = state_dict.pop('transformer.embeddings.position_embeddings.weight')
    state_dict['transformer.embeddings.position_embeddings.weight'] = pos_embeddings[2:]
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = (math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r'^transformer.final_layer_norm.', r'transformer.ln_f.', key)
        # The OPT-175B checkpoint calls this 'decoder.layer_norm' instead of 'decoder.final_layer_norm'
        key = re.sub(r'^transformer.layer_norm.', r'transformer.ln_f.', key)
        key = re.sub(r'^transformer.layers.(\d+).self_attn_layer_norm.',
                     r'transformer.layers.\1.norm1.', key)
        key = re.sub(r'^transformer.layers.(\d+).final_layer_norm.',
                     r'transformer.layers.\1.norm2.', key)
        return key
    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        return re.sub(r'^transformer.layers.(\d+).fc(1|2).',
                      r'transformer.layers.\1.mlp.fc\2.', key)
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f'transformer.layers.{l}.self_attn.q_proj.weight')
        Wk = state_dict.pop(f'transformer.layers.{l}.self_attn.k_proj.weight')
        Wv = state_dict.pop(f'transformer.layers.{l}.self_attn.v_proj.weight')
        bq = state_dict.pop(f'transformer.layers.{l}.self_attn.q_proj.bias')
        bk = state_dict.pop(f'transformer.layers.{l}.self_attn.k_proj.bias')
        bv = state_dict.pop(f'transformer.layers.{l}.self_attn.v_proj.bias')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat(
            [Wq, Wk, Wv], dim=0
        )
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.bias'] = torch.cat(
            [bq, bk, bv], dim=0
        )
    def key_mapping_attn(key):
        return re.sub(r'^transformer.layers.(\d+).self_attn.out_proj.',
                      r'transformer.layers.\1.mixer.out_proj.', key)
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def gptj_config_to_gpt2_config(gptj_config: GPTJConfig) -> GPT2Config:
    headdim = gptj_config.n_embd // gptj_config.n_head
    return GPT2Config(
        vocab_size=gptj_config.vocab_size,
        n_positions=gptj_config.n_positions,
        n_embd=gptj_config.n_embd,
        n_layer=gptj_config.n_layer,
        n_head=gptj_config.n_head,
        n_inner=gptj_config.n_inner,
        activation_function=gptj_config.activation_function,
        resid_pdrop=gptj_config.resid_pdrop,
        embd_pdrop=gptj_config.embd_pdrop,
        attn_pdrop=gptj_config.attn_pdrop,
        layer_norm_epsilon=gptj_config.layer_norm_epsilon,
        initializer_range=gptj_config.initializer_range,
        bos_token_id=gptj_config.bos_token_id,
        eos_token_id=gptj_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        prenorm=True,
        rotary_emb_fraction=gptj_config.rotary_dim / headdim,
        rotary_emb_interleaved=True
    )
