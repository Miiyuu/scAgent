# %%
# %%
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

# sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator, MultiTransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

import loralib as lora

from config import MODEL_PATHS

logger = scg.logger

def load_weight(model, model_file, lora_file, model_type):
    model_dict = model.state_dict(keep_vars=True)
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)

    if lora_file is not None:
        print("Load LoRA...")
        lora_dict = torch.load(lora_file)
        lora_dict = {
            k: v
            for k, v in lora_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(lora_dict)
    else:
        print("LoRA file not provided, disabling non-pretrained layers...")
        freeze_cnt = 0
        for name, param in model.named_parameters():
            if name not in pretrained_dict:
                param.data.zero_()  # 确保未加载的参数不会影响计算
                param.requires_grad = False  # 冻结未加载的参数，防止训练修改
                freeze_cnt += 1
        print(f"Freeze {freeze_cnt} parameters")

    if model_type == 'cell':
        print("Load cls...")
        cls_file = MODEL_PATHS["cls_file"]
        cls_dict = torch.load(cls_file)
        model_dict.update(cls_dict)

    model.load_state_dict(model_dict)
    return model


def load_multi_weight(model, model_file, lora_folder, model_type='cell'):
    # lora model
    # lora.mark_only_lora_as_trainable(model)

    # only load params that are in the model and match the size
    model_dict = model.state_dict(keep_vars=True)
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    # # lora_dict = torch.load("../ts_save/add_intestine/lora_model.pt")
    # lora_dict = torch.load("../ts_save_bn+lora/ts_small_intestine-Apr17-10-12/lora_model.pt")
    model_dict.update(pretrained_dict)

    for sub_dir in os.listdir(lora_folder):
        if sub_dir.startswith("tissue"):
            sub_dir_path = os.path.join(lora_folder, sub_dir)
            lora_model_path = os.path.join(sub_dir_path, "lora_model.pt")
            lora_key = sub_dir.removeprefix("tissue")
            if os.path.exists(lora_model_path):
                lora_dict = torch.load(lora_model_path)

                new_lora_dict = {}
                for k, v in lora_dict.items():
                    if "lora_" in k:
                        parts = k.split("Wqkv.")
                        if len(parts) == 2:
                            new_k = f"{parts[0]}Wqkv.moelora_modules.sctab_{lora_key}.{parts[1]}"
                        else:
                            new_k = k
                    elif "bn." in k:
                        parts = k.split("bn.")
                        if len(parts) == 2:
                            new_k = f"{parts[0]}bn_modules.sctab_{lora_key}.{parts[1]}"
                        else:
                            new_k = k
                    else:
                        new_k = k
                    new_lora_dict[new_k] = v

                for k, v in new_lora_dict.items():
                    logger.info(f"hahaha Loading params {k} with shape {v.shape}")

                lora_dict = {
                    k: v
                    for k, v in new_lora_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }

                model_dict.update(lora_dict)
                model.load_state_dict(model_dict)
            else:
                logger.warning(f"LoRA model not found in {sub_dir_path}")

    if model_type == 'cell':
        print("Load cls...")
        cls_file = MODEL_PATHS["cls_file"]

        cls_dict = torch.load(cls_file)

        model_dict.update(cls_dict)

    model.load_state_dict(model_dict)

    # my_print_trainable_parameters(model)

    return model


def initModel(parameter, data_type, cls_mode):

    # paradict=['ntokens','embsize','nhead','d_hid','nlayers',
    #           'vocab','dropout','pad_token',
    #           'pad_value','MVC','DAB','INPUT_BATCH_LABELS','num_batch_types',
    #           'DSBN','input_emb_style','n_input_bins','cell_emb_style','mvc_decoder_style',
    #           'ecs_thres','explicit_zero_prob','fast_transformer','fast_transformer_backend','pre_norm']
    
    # for para in paradict:
    #     print(parameter[para])
    n_cls = 200
    if cls_mode == 'cell':
        n_cls = 200
    elif cls_mode == 'tissue':
        if data_type == 'ts':
            n_cls = 24
        elif data_type == 'sctab':
            n_cls = 35


    model = TransformerModel(
        parameter['ntokens'],
        parameter['embsize'],
        parameter['nhead'],
        parameter['d_hid'],
        parameter['nlayers'],
        nlayers_cls=3,
        # n_cls=parameter['num_types'] if parameter['CLS'] else 1,
        n_cls=n_cls,
        vocab=parameter['vocab'],
        dropout=parameter['dropout'],
        pad_token=parameter['pad_token'],
        pad_value=parameter['pad_value'],
        do_mvc=parameter['MVC'],
        do_dab=parameter['DAB'],
        use_batch_labels=parameter['INPUT_BATCH_LABELS'],
        num_batch_labels=parameter['num_batch_types'],
        domain_spec_batchnorm=parameter['DSBN'],
        input_emb_style=parameter['input_emb_style'],
        n_input_bins=parameter['n_input_bins'],
        cell_emb_style=parameter['cell_emb_style'],
        mvc_decoder_style=parameter['mvc_decoder_style'],
        ecs_threshold=parameter['ecs_thres'],
        explicit_zero_prob=parameter['explicit_zero_prob'],
        use_fast_transformer=parameter['fast_transformer'],
        fast_transformer_backend=parameter['fast_transformer_backend'],
        pre_norm=parameter['pre_norm'],
        islora=parameter['islora'],
    )

    # for name, para in model.named_parameters():
    #     print("-" * 20)
    #     print(f"name: {name}")

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")


    return model


def initMultiModel(parameter, lora_keys, data_type, cls_mode):
    # paradict=['ntokens','embsize','nhead','d_hid','nlayers',
    #           'vocab','dropout','pad_token',
    #           'pad_value','MVC','DAB','INPUT_BATCH_LABELS','num_batch_types',
    #           'DSBN','input_emb_style','n_input_bins','cell_emb_style','mvc_decoder_style',
    #           'ecs_thres','explicit_zero_prob','fast_transformer','fast_transformer_backend','pre_norm']

    # for para in paradict:
    #     print(parameter[para])

    n_cls = 200
    if cls_mode == 'cell':
        n_cls = 200
    elif cls_mode == 'tissue':
        if data_type == 'ts':
            n_cls = 24
        elif data_type == 'sctab':
            n_cls = 35


    model = MultiTransformerModel(
        parameter['ntokens'],
        parameter['embsize'],
        parameter['nhead'],
        parameter['d_hid'],
        parameter['nlayers'],
        nlayers_cls=3,
        # n_cls=parameter['num_types'] if parameter['CLS'] else 1,
        n_cls=n_cls,
        vocab=parameter['vocab'],
        dropout=parameter['dropout'],
        pad_token=parameter['pad_token'],
        pad_value=parameter['pad_value'],
        do_mvc=parameter['MVC'],
        do_dab=parameter['DAB'],
        use_batch_labels=parameter['INPUT_BATCH_LABELS'],
        num_batch_labels=parameter['num_batch_types'],
        domain_spec_batchnorm=parameter['DSBN'],
        input_emb_style=parameter['input_emb_style'],
        n_input_bins=parameter['n_input_bins'],
        cell_emb_style=parameter['cell_emb_style'],
        mvc_decoder_style=parameter['mvc_decoder_style'],
        ecs_threshold=parameter['ecs_thres'],
        explicit_zero_prob=parameter['explicit_zero_prob'],
        use_fast_transformer=parameter['fast_transformer'],
        fast_transformer_backend=parameter['fast_transformer_backend'],
        pre_norm=parameter['pre_norm'],
        islora=parameter['islora'],
        lora_keys=lora_keys
    )

    # for name, para in model.named_parameters():
    #     print("-" * 20)
    #     print(f"name: {name}")

    post_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")

    return model

def my_print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        logger.info(f"Loading params {name} with shape {param.shape}, device:{param.device}")

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



