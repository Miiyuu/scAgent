import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys

# sys.path.insert(0, "../")
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
import scgpt as scg
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics


import loralib as lora

import tool.build_model as buildModel
import tool.data_process as dataProcess

# from memory.search_db import query_cell,query_tissue

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from tqdm import tqdm

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from config import MODEL_PATHS, DICT_PATHS





class ScgptModel:
    def __init__(self, input_data=None,data_type="sctab",islora=True, lora_folder=None):

        self.input_data=input_data

        self.data_type=data_type

        self.islora=islora
        self.lora_folder = lora_folder

        
    def build_model(self,model_type,lora_file=None,gene_idx=None,batch_size=32):
        
        if model_type=="tissue":
            if self.data_type == "sctab":
                lora_file=MODEL_PATHS["tissue_lora_model"]
            elif self.data_type == "ts":
                lora_file=MODEL_PATHS["tissue_lora_model_ts"]

        elif model_type=="cell":

            # 打开并读取 JSON 文件
            with open(DICT_PATHS["id2type"], 'r') as file:
                id2type = json.load(file)

            self.id2type=id2type

        if gene_idx!=None:
            # 打开文件并读取数据
            self.gene_idx = gene_idx


        parameter = dict(
            seed=0,
            load_model=MODEL_PATHS["pretrained"],
            lora_file=lora_file,
            mask_ratio=0.0,
            MVC=False,  # Masked value prediction for cell embedding
            ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
            dab_weight=0.0,
            batch_size=batch_size,
            layer_size=128,
            n_bins=51,
            nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.2,  # dropout probability
            fast_transformer=True,
            # fast_transformer=False,
            pre_norm=False,
            amp=True,  # Automatic Mixed Precision
            include_zero_gene=False,
            DSBN=False,  # Domain-spec batchnorm
        )

        parameter['islora']=self.islora


        self.add_Modelconfig(parameter)

        parameter['num_batch_types'] = 1

        parameter['ntokens'] = len(parameter['vocab'])  # size of vocabulary

        if self.lora_folder:
            lora_keys = self.get_lora_keys()
            model = buildModel.initMultiModel(parameter, cls_mode=model_type, lora_keys=lora_keys, data_type=self.data_type)
            model.to(device)
            model = buildModel.load_multi_weight(model, parameter['model_file'], model_type=model_type, lora_folder=self.lora_folder)
        else:
            model = buildModel.initModel(parameter, cls_mode=model_type, data_type=self.data_type)
            model.to(device)
            model = buildModel.load_weight(model, parameter['model_file'], parameter['lora_file'],model_type=model_type)

        self.parameter=parameter
        self.model=model
        self.model_type=model_type


    def data_process(self,use_idx=False,data_type="test",drop_last=False):

        adata=self.input_data
        
        processed_adata=dataProcess.process(adata,self.parameter['vocab'],datatype=self.data_type)

        self.parameter['genes'] = processed_adata.var["gene_name"].tolist()

        self.parameter['gene_ids'] = np.array(self.parameter['vocab'](self.parameter['genes']), dtype=int)

        batch_ids = processed_adata.obs["batch_id"].tolist()
        self.parameter['num_batch_types'] = len(set(batch_ids))
        self.parameter['batch_ids'] = np.array(batch_ids)

        self.adata=processed_adata

        parameter=self.parameter

        all_counts = (
            processed_adata.layers[parameter['input_layer_key']].A
            if issparse(processed_adata.layers[parameter['input_layer_key']])
            else processed_adata.layers[parameter['input_layer_key']]
        )

        celltypes_labels = processed_adata.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)

        self.celltypes_labels=celltypes_labels  #保存数据集的标签

        batch_ids = processed_adata.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        
        gene_idx=None
        if use_idx and self.gene_idx!=None:
            gene_idx=self.gene_idx
        
        tokenized_test = tokenize_and_pad_batch(
            all_counts,
            parameter['gene_ids'],
            max_len=parameter['max_seq_len'],
            vocab=parameter['vocab'],
            pad_token=parameter['pad_token'],
            pad_value=parameter['pad_value'],
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=parameter['include_zero_gene'],
            # include_zero_gene=True,
            gene_idx=gene_idx,
            use_idx=use_idx
        )

        input_values_test = random_mask_value(
            tokenized_test["values"],
            mask_ratio=parameter['mask_ratio'],
            mask_value=parameter['mask_value'],
            pad_value=parameter['pad_value'],
        )

        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels).long(),
        }


        if data_type=="test":

            test_loader = DataLoader(
                dataset=TestSeqDataset(test_data_pt),
                batch_size=parameter['batch_size'],
                shuffle=False,
                drop_last=drop_last,
                # num_workers=min(len(os.sched_getaffinity(0)), parameter['batch_size'] // 2),
                num_workers=0,
                pin_memory=True,
            )
        else:
            test_loader = DataLoader(
                dataset=TrainSeqDataset(test_data_pt,parameter['batch_size']),
                batch_size=parameter['batch_size'],
                shuffle=False,
                drop_last=False,
                # num_workers=min(len(os.sched_getaffinity(0)), parameter['batch_size'] // 2),
                num_workers=0,
                pin_memory=True,
            )


        self.data_loader=test_loader




    def add_Modelconfig(self,parameter):

        # settings for input and preprocessing
        pad_token = "<pad>"
        parameter['pad_token']="<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        set_seed(parameter['seed'])
        parameter['mask_value'] = "auto"  # for masked values, now it should always be auto

        parameter['max_seq_len'] = 3001

        model_dir = Path(parameter['load_model'])
        model_config_file = model_dir / "args.json"
        parameter['model_file'] = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)

        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        # logger.info(
        #     f"Resume model from {model_file}, the model args will override the "
        #     f"config {model_config_file}."
        # )
        parameter['embsize'] = model_configs["embsize"]
        parameter['nhead'] = model_configs["nheads"]
        parameter['d_hid'] = model_configs["d_hid"]
        parameter['nlayers'] = model_configs["nlayers"]
        parameter['n_layers_cls'] = model_configs["n_layers_cls"]

        # settings for training
        MLM = False  # whether to use masked language modeling, currently it is always on.
        parameter['CLS'] = True  # celltype classification objective
        ADV = False  # Adversarial training for batch correction
        DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
        parameter['DAB']=DAB
        parameter['INPUT_BATCH_LABELS'] = False  # TODO: have these help MLM and MVC, while not to classifier
        input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
        parameter['input_emb_style']="continuous"
        parameter['cell_emb_style'] = "cls"  # "avg-pool" or "w-pool" or "cls"
        parameter['mvc_decoder_style'] = "inner product"


        explicit_zero_prob = MLM and parameter['include_zero_gene']  # whether explicit bernoulli for zeros
        parameter['do_sample_in_train'] = False and explicit_zero_prob  # sample the bernoulli in training
        parameter['explicit_zero_prob']=explicit_zero_prob

        # settings for the model
        parameter['fast_transformer_backend'] = "flash"  # "linear" or "flash"

        # input/output representation
        input_style = "binned"  # "normed_raw", "log1p", or "binned"

        # %% validate settings
        assert input_style in ["normed_raw", "log1p", "binned"]
        assert input_emb_style in ["category", "continuous", "scaling"]
        if input_style == "binned":
            if input_emb_style == "scaling":
                raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif input_style == "log1p" or input_style == "normed_raw":
            if input_emb_style == "category":
                raise ValueError(
                    "input_emb_style `category` is not supported for log1p or normed_raw input."
                )

        if input_emb_style == "category":
            parameter['mask_value'] = parameter['n_bins'] + 1
            parameter['pad_value'] = parameter['n_bins']  # for padding gene expr values
            parameter['n_input_bins'] = parameter['n_bins'] + 2
        else:
            parameter['mask_value'] = -1
            parameter['pad_value'] = -2
            parameter['n_input_bins'] = parameter['n_bins']

        if ADV and DAB:
            raise ValueError("ADV and DAB cannot be both True.")

        vocab.set_default_index(vocab["<pad>"])
        parameter['vocab']=vocab
        # %%
        parameter['input_layer_key'] = {  # the values of this map coorespond to the keys in preprocessing
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[input_style]


        # %% inference


    def inference(self,mode="cls"):
        """
        Evaluate the model on the evaluation data.
        """

        model=self.model
        loader=self.data_loader
        parameter=self.parameter

        model.eval()
        cell_embeddings=[]
        predictions=[]
        true_labels=[]  
        with torch.no_grad():
            for batch_data in tqdm(loader, desc="embedding"):
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)
                true_labels.extend(celltype_labels.cpu().numpy().tolist())

                # print(parameter['vocab']['<pad>'])

                src_key_padding_mask = input_gene_ids.eq(parameter['vocab']['<pad>'])
                with torch.cuda.amp.autocast(enabled=parameter['amp']):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if parameter['INPUT_BATCH_LABELS'] or parameter['DSBN'] else None,
                        CLS=parameter['CLS'],  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=parameter['do_sample_in_train'],
                        # generative_training = False,
                    )

                    output_emb=output_dict["cell_emb"]
                    output_values = output_dict["cls_output"]
                
                output=output_emb.squeeze().cpu().numpy()

                preds = output_values.argmax(1).cpu().numpy()

                predictions.extend(preds)


                if output.ndim==1:
                    cell_embeddings.append(output)
                else:
                    cell_embeddings.extend(output)
        

        if self.model_type=="cell" and self.data_type=="sctab":
            id2type=self.id2type
            predictions = [id2type.get(str(pred), "Unknown") for pred in predictions]

        self.true_labels=true_labels
        self.cell_embeddings=cell_embeddings
        self.predictions=predictions

        if mode=="cls":

            return predictions

        if mode=="both":
            return cell_embeddings, predictions
        
        elif mode=="emb":
            return cell_embeddings

    def multi_inference(self, data_lora_keys, mode="cls",  is_rag=False):
        """
        Evaluate the model on the evaluation data.
        """

        model = self.model
        loader = self.data_loader
        parameter = self.parameter

        model.eval()
        cell_embeddings = []
        predictions = []
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(loader, desc="embedding")):
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)

                # print(parameter['vocab']['<pad>'])

                src_key_padding_mask = input_gene_ids.eq(parameter['vocab']['<pad>'])
                with torch.cuda.amp.autocast(enabled=parameter['amp']):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if parameter['INPUT_BATCH_LABELS'] or parameter['DSBN'] else None,
                        CLS=parameter['CLS'],  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=parameter['do_sample_in_train'],
                        # generative_training = False,
                        lora_key=data_lora_keys[i]
                    )

                    output_emb = output_dict["cell_emb"]
                    output_values = output_dict["cls_output"]

                output = output_emb.squeeze().cpu().numpy()

                preds = output_values.argmax(1).cpu().numpy()

                predictions.extend(preds)

                if output.ndim == 1:
                    cell_embeddings.append(output)
                else:
                    cell_embeddings.extend(output)

        if self.model_type == "cell" and self.data_type == "sctab":
            id2type = self.id2type
            predictions = [id2type.get(str(pred), "Unknown") for pred in predictions]

        if mode == "cls":

            if is_rag:
                rag_result = self.check_with_RAG(cell_embeddings, mode=self.model_type)

                return (predictions, rag_result)
            else:
                return predictions

        if mode == "both":
            return cell_embeddings, predictions

        elif mode == "emb":
            return cell_embeddings

    # def check_with_RAG(self,embeddings,mode="cell"):


    #     if mode=="cell":
    #         return query_cell(embeddings)
    #     elif mode=="tissue":
    #         return query_tissue(embeddings)

    def get_lora_keys(self):
        lora_keys = []
        for subdir in os.listdir(self.lora_folder):
            if subdir.startswith("tissue"):
                # lora_model_path = os.path.join(lora_folder, subdir, "lora_model.pt")
                key = subdir.removeprefix("tissue")
                lora_keys.append("sctab_" + key)

        return lora_keys




# dataset
class TestSeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict

class TrainSeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = self.data["gene_ids"].shape[0]

    def __len__(self):
        return ((self.num_samples + self.batch_size - 1) // self.batch_size) * self.batch_size

    def __getitem__(self, idx):
        batch_start = (idx // self.batch_size) * self.batch_size
        batch_indices = list(range(batch_start, min(batch_start + self.batch_size, self.num_samples)))

        # 填充不足 batch_size 的部分
        while len(batch_indices) < self.batch_size:
            batch_indices.append(batch_indices[-1])

        current_idx = idx % self.batch_size
        selected_idx = batch_indices[current_idx]

        return {k: v[selected_idx] for k, v in self.data.items()}
    
