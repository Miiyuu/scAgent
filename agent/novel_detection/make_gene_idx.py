import sys
import scanpy as sc
from pathlib import Path
import pandas as pd
import anndata as ad
from scgpt.preprocess import Preprocessor
import numpy as np
import scgpt as scg

import glob
import os

from scgpt.tokenizer.gene_tokenizer import GeneVocab

from scipy.sparse import issparse

import pickle

import json

import re

from config import MODEL_PATHS, DICT_PATHS

logger = scg.logger

data_is_raw=False
filter_gene_by_counts=False
n_bins=51
even_binning=False


data_is_raw=False
filter_gene_by_counts=False
n_bins=51
even_binning=False



def make_high_exp_gene(data_path):
    
    adata = sc.read(data_path)
    
    adata.obs["str_batch"] = "train"

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels

    adata.var["gene_name"] = adata.var.index.tolist()

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab_file=MODEL_PATHS["pretrained"]+"/vocab.json"
    vocab = GeneVocab.from_file(vocab_file)

    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    vocab.set_default_index(vocab["<pad>"])

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]

    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    # 剔除不在词表中的基因
    adata = adata[:, adata.var["id_in_vocab"] >= 0]


    # %%
    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        # normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        normalize_total=False,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        even_binning=False,
    )

    preprocessor(adata, batch_key=None)
    input_style = "binned"
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]
    data = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # 初始化一个空列表来存储每个行的交集
    intersections = []
    print(len(data))

    total_length=len(data)

    for i in range(len(data)):
        row = data[i]
        indices = np.where(row != -3)[0]
        idx = np.nonzero(row)[0]
        
        # 找到 indices 和 idx 的交集
        x = np.intersect1d(indices, idx)
        
        # 将交集添加到 intersections 列表中
        intersections.append(x)

    # 将 intersections 列表展平成一个一维数组
    flattened_intersections = np.concatenate(intersections)

    # 统计每个值的数量
    unique_values, counts = np.unique(flattened_intersections, return_counts=True)

    # 获取数量从大到小排序的索引
    sorted_indices = np.argsort(counts)[::-1]

    # 根据排序后的索引重新排列 unique_values 和 counts
    sorted_unique_values = unique_values[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # 选取 sorted_unique_values 中的排名前 3000 个值，构成列表
    top_3000_genes = sorted_unique_values[:3000].tolist()


    top_3000_counts = sorted_counts[:3000].tolist()

    # 打印排名前 3000 的值及其对应的数量，仅打印 count > total_length * 0.9 的值
    high_exp_gene=[]
    print("Top 3000 genes and their counts (count > 60% of total_length):")
    for gene, count in zip(top_3000_genes, top_3000_counts):
        # print(f"Gene: {gene}, Count: {count}/{total_length}")
        if count > total_length * 0.6:
            # print(f"Gene: {gene}, Count: {count}/{total_length}")
            high_exp_gene.append(gene)


    print("total:",len(high_exp_gene))

    #对top_3000_genes进行从小到大排序
    high_exp_gene.sort()

    gene_names = adata.var_names[high_exp_gene].tolist()


    return gene_names


def make_common_geneidx(adata,ref_genes):

    raw_genes=adata.var_names


    # 计算 raw_genes 和 ref_genes 的交集
    common_genes = set(raw_genes).intersection(set(ref_genes))
    

    # # 输出结果
    # print(f"cancer_genes 和 tab_genes 的交集有 {len(common_genes)} 个基因。")
    # print("交集基因列表：")
    # print(common_genes)

    # 如果交集基因数量不足 3000，补充到 3000
    if len(common_genes) < 3000:
        # 计算 cancer_genes 中每个基因的平均表达值
        gene_mean_expression = adata.X.mean(axis=0).A1  # 计算平均表达值

        # 将基因名称与平均表达值对应
        gene_expression_dict = dict(zip(raw_genes, gene_mean_expression))

        # 过滤掉已经在交集中的基因
        remaining_genes = [gene for gene in raw_genes if gene not in common_genes]

        # 按平均表达值从高到低排序
        sorted_remaining_genes = sorted(remaining_genes, key=lambda x: gene_expression_dict[x], reverse=True)

        # 选取 top (3000 - len(common_genes)) 个基因
        additional_genes = sorted_remaining_genes[:3000 - len(common_genes)]

        # 将补充的基因加入交集
        final_genes = list(common_genes) + additional_genes

        # 输出结果
        print(f"补充后的基因数量为 {len(final_genes)} 个基因。")
        # print("补充后的基因列表：")
        # print(final_genes)
    else:
        print("交集基因数量已达到 3000，无需补充。")


    
    # 找到 final_genes 在 cancer_genes 中的索引
    final_gene_indices = [raw_genes.tolist().index(gene) for gene in final_genes]

    # print(final_gene_indices)

    return final_gene_indices


def get_gene_idx(ref_data_path,adata):


    ref_genes=make_high_exp_gene(ref_data_path)


    final_gene_idx=make_common_geneidx(adata,ref_genes)

    return final_gene_idx



