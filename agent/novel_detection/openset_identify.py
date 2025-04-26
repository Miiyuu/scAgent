import sys
import re
import os
# 将项目的根目录添加到 sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import scanpy as sc

import pickle

from tool.scgpt_model import ScgptModel


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd

import numpy as np

from .make_gene_idx import get_gene_idx

from .process_data import fliter_gene_in_common,split_normal_and_abnormal

from .calculate_dis import statistic_dis_lora,statistic_dis_nolora

from .LLModel import DeepSeek,build_prompt

from scAgent.memory.search_db import query

from config import MODEL_PATHS, DATA_PATHS  



def make_embedding(lora_file,adata,gene_idx=None,use_idx=False,islora=True):


    scgptmodel = ScgptModel(adata,data_type="sctab",islora=islora)

    scgptmodel.build_model(model_type="cell",lora_file=lora_file,gene_idx=gene_idx)

    scgptmodel.data_process(use_idx=use_idx)

    cell_embeddings=scgptmodel.inference(mode="emb")

    # embed_dict={}
    # for i,label in enumerate(celltypes_labels):
    #     if label not in embed_dict:
    #         embed_dict[label]=[cell_embeddings[i]]
    #     else:
    #         embed_dict[label].append(cell_embeddings[i])

    return cell_embeddings

def identify(ref_data_path,test_adata,tissue_id,tissue_name):


    gene_idx=get_gene_idx(ref_data_path,test_adata)

    # gene_dix_path="/data/lpg/codes/scAgent/lpg/openClassifier/cancer_data/normal/liver/gene_idx.pkl"
    
    # # 读取 pickle 文件
    # with open(gene_dix_path, 'rb') as file:
    #     gene_idx = pickle.load(file)

    # print(gene_idx)

    
    lora_dir=MODEL_PATHS["lora_dir"]

    lora_file=f"{lora_dir}/{tissue_id}/lora_model.pt"

    cell_embs=make_embedding(lora_file,test_adata,gene_idx,use_idx=True,islora=True)

    nolora_cell_embs=make_embedding(None,test_adata,gene_idx,use_idx=True,islora=False)

    distanceslist_lora=statistic_dis_lora(tissue_name,cell_embs)

    distanceslist_nolora=statistic_dis_nolora(tissue_name,nolora_cell_embs)

    LLModel=DeepSeek()

    predict_celltypes=[]
    explanations=[]

    for i in range(len(distanceslist_lora)):

        distances_lora=distanceslist_lora[i]

        distances_nolora=distanceslist_nolora[i]

        prompt=build_prompt(distances_lora,distances_nolora)

        res=LLModel.model_generate(prompt)

        predict_celltype,explanation=LLModel.process_output(res)

        # 检查是否返回了错误
        if predict_celltype == "error":
            print(f"Error processing output: {explanation}")
            # 可以选择跳过这个结果，或者根据需求处理错误
            continue  # 跳过当前循环，继续处理下一个 res

        predict_celltypes.append(predict_celltype)
        explanations.append(explanation)

    return predict_celltypes,explanations


def openset_recognize(tissue_id,tissue_name,raw_adata):


    ref_data_path=f"{DATA_PATHS['sctab_data']}/{tissue_id}/train.h5ad"

    ref_adata=sc.read(ref_data_path)

    filtered_adata=fliter_gene_in_common(raw_adata,ref_adata)

    return identify(ref_data_path,filtered_adata,tissue_id,tissue_name)


def novel_detect(state):

    tissue_name=state["tissue_name"]

    tissue_id=f"tissue{state['tissue_id']}"

    raw_data_path=state["file_path"]

    raw_adata=sc.read(raw_data_path)

    # ref_data_path=f"/data2/lpg_data/sctab_data/{tissue_id}/train.h5ad"

    # ref_adata=sc.read(ref_data_path)

    # 可选,划分normal和abnormal的细胞
    # other_adata,malignant_adata=split_normal_and_abnormal(raw_adata,ref_adata)

    # malignant_adata=malignant_adata[:1000]

    predict_celltypes,explanations=openset_recognize(tissue_id,tissue_name,raw_adata)

    update_state={"response":f"{predict_celltypes}\n{explanations}"}

    return update_state


    

