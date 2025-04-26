import sys
import re
import os
# 将项目的根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scgptTools.scgptModel import ScgptModel
from pathlib import Path

import scanpy as sc

import pickle

import pandas as pd

import numpy as np
import scipy.sparse as sp

from scipy.sparse import issparse


from scgpt.tokenizer.gene_tokenizer import GeneVocab

from scgpt.preprocess import Preprocessor

import json

from config import MODEL_PATHS, DICT_PATHS

model_dir = Path(MODEL_PATHS["pretrained"])

vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)

def fliter_gene_in_vocab(adata):

    adata.var["gene_name"] = adata.var.index.tolist()

    # 剔除不在词表中的基因
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]

    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes ")
    print(f"in vocabulary of size {len(vocab)}.")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    return adata

def fliter_gene_in_common(raw_adata,ref_adata):
    

    # 将 var_names（基因ID）转换为基因名称
    # train_adata.var_names = train_adata.var.gene_symbols

    if 'feature_name' in raw_adata.var:
        raw_adata.var_names = raw_adata.var.feature_name

    raw_adata=fliter_gene_in_vocab(raw_adata)

    # 打印基因名字
    new_gene_names = raw_adata.var_names  # 获取基因名字
    print(len(new_gene_names))

    # ts_adata_path=f"/data/lpg/codes/scAgent/share/ts_data/ts_{tissue_name}/test.h5ad"

    ref_adata=fliter_gene_in_vocab(ref_adata)

    ref_gene_names = ref_adata.var_names  # 获取基因名字
    print(len(ref_gene_names))

    # 计算 new_gene_names 中有多少在 ts_gene_names 中
    common_genes = set(new_gene_names).intersection(set(ref_gene_names))
    print(f"new_gene_names 中有 {len(common_genes)} 个基因在 ref_gene_names 中。")

    # 筛选出在 ts_gene_names 中存在的基因，并按照 ts_gene_names 的顺序排列
    filtered_gene_names = [gene for gene in ref_gene_names if gene in common_genes]

    filtered_adata = raw_adata[:, filtered_gene_names].copy()

    return filtered_adata

def split_normal_and_abnormal(raw_adata,ref_adata):

    # 提取cell_ontology_class列
    cell_ontology_class = raw_adata.obs['cell_type']

    # 2. 分离 malignant cell 和其余细胞
    is_malignant = (cell_ontology_class == "malignant cell") | (cell_ontology_class == "abnormal cell")

    # 3. 创建两个新的 AnnData 对象
    # 3.1 提取 malignant cell 的 AnnData 对象
    malignant_adata = raw_adata[is_malignant, :].copy()

    # 3.2 提取其余细胞的 AnnData 对象
    other_adata = raw_adata[~is_malignant, :].copy()

    malignant_adata.obs["cell_type"] = 164

    print("恶性细胞数量：",malignant_adata.n_obs)

    # 2. 将 other_adata 的 cell_type 转换为小写并进行映射
    # 2.1 将 cell_type 转换为小写
    other_adata.obs["cell_type"] = other_adata.obs["cell_type"].apply(lambda x: x.lower() if isinstance(x, str) else x)

    cell_id2name_path=DICT_PATHS["cell_id2name"]

    # 读取 JSON 文件
    with open(cell_id2name_path, 'r') as file:
        data = json.load(file)

    # 将值转换为小写
    cell_id2name = {key: value.lower() for key, value in data.items()}

    # 反转字典，将细胞名称作为键，ID 作为值
    name2id = {value: key for key, value in cell_id2name.items()}

    # 将 cell_type 映射到索引
    other_adata.obs["cell_type"] = other_adata.obs["cell_type"].map(name2id)

    # 过滤掉 cell_type 为 NaN 的细胞
    other_adata = other_adata[~other_adata.obs["cell_type"].isna()]

    # 将映射后的 ID 转换为 int 类型
    other_adata.obs["cell_type"] = other_adata.obs["cell_type"].astype(int)

    malignant_adata.obs["celltype"] = malignant_adata.obs["cell_type"].astype("category")

    other_adata.obs["celltype"] = other_adata.obs["cell_type"].astype("category")


    print(other_adata.obs["celltype"].cat.categories)

    ref_adata.obs["celltype"] = ref_adata.obs["cell_type"].astype("category")

    # 提取细胞类型的集合
    cell_type_set = ref_adata.obs["celltype"].cat.categories

    # 打印细胞类型的集合
    print(cell_type_set)

    # 2. 过滤 other_adata 中的 cell_type
    # 2.1 判断 other_adata 中的 cell_type 是否在 cell_type_set 中
    is_valid_cell_type = other_adata.obs["cell_type"].isin(cell_type_set)

    # 2.2 保留在 cell_type_set 中的细胞
    other_adata = other_adata[is_valid_cell_type, :].copy()

    # 3. 更新 other_adata 的 celltype 为 category 类型
    other_adata.obs["celltype"] = other_adata.obs["cell_type"].astype("category")

    print(other_adata.obs["celltype"].cat.categories)

    other_adata.obs["str_batch"] = f"train"

    malignant_adata.obs["str_batch"] = f"train"

    return other_adata,malignant_adata

# if __name__ == '__main__':
     

#     tissue_name="liver"

#     tissue_id="tissue25"


#     adata_path=f"/data/lpg/codes/scAgent/share/agent/openClassifier/cancer_data/{tissue_name}.h5ad"


#     ref_adata_path=f"/data/lpg/codes/scAgent/share/sctab_data/{tissue_id}/test.h5ad"

#     raw_adata=sc.read(adata_path)

#     ref_adata=sc.read(ref_adata_path)


#     filtered_adata=fliter_gene_in_common(raw_adata,ref_adata)


#     other_adata,malignant_adata=split_normal_and_abnormal(filtered_adata,ref_adata)


#     # # 定义保存路径
#     # normal_path = f"/data/lpg/codes/scAgent/share/agent/openClassifier/cancer_data/tissue_data2/normal/{tissue_name}"
#     # abnormal_path = f"/data/lpg/codes/scAgent/share/agent/openClassifier/cancer_data/tissue_data2/abnormal/{tissue_name}"

#     # 定义保存路径
#     normal_path = f"/data/lpg/codes/scAgent/share/agent/openClassifier/cancer_data/normal/{tissue_name}"
#     abnormal_path = f"/data/lpg/codes/scAgent/share/agent/openClassifier/cancer_data/abnormal/{tissue_name}"

#     # 确保目标文件夹存在，如果不存在则创建
#     os.makedirs(normal_path, exist_ok=True)
#     os.makedirs(abnormal_path, exist_ok=True)

#     other_adata.write(f"{normal_path}/{tissue_name}.h5ad")

#     malignant_adata.write(f"{abnormal_path}/{tissue_name}.h5ad")

