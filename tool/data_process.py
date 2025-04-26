import sys
# sys.path.insert(0, "../")
import scanpy as sc
from pathlib import Path
import anndata as ad
import os
print(os.path.abspath(__file__))

from scgpt.preprocess import Preprocessor
import numpy as np
import scgpt as scg
import random


logger = scg.logger

data_is_raw=False
filter_gene_by_counts=False
n_bins=51
even_binning=False

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
    # even_binning=even_binning,
)

def process(adata, vocab, datatype):
    if datatype == "sctab":
        # make the batch category column
        if "cell_type" in adata.obs.columns and "celltype" not in adata.obs.columns:
            adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata.obs["str_batch"] = f"test"
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        # celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltype_id_labels = adata.obs["celltype"]

        adata.obs["celltype_id"] = celltype_id_labels

        adata.var["gene_name"] = adata.var.index.tolist()

        # 剔除不在词表中的基因
        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        preprocessor(adata, batch_key=None)

        return adata
    elif datatype == "ts":
        # make the batch category column
        if "cell_type" in adata.obs.columns and "celltype" not in adata.obs.columns:
            adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        if "cell_ontology_class" in adata.obs.columns and "celltype" not in adata.obs.columns:
            adata.obs["celltype"] = adata.obs["cell_ontology_class"].astype("category")
        adata.obs["str_batch"] = f"test"
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        # celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltype_id_labels = adata.obs["celltype"]

        adata.obs["celltype_id"] = celltype_id_labels

        adata.var["gene_name"] = adata.var.index.tolist()

        # 剔除不在词表中的基因
        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        preprocessor(adata, batch_key=None)

        return adata

