import json
import operator
import os
import random
import re
import sys
from typing import Annotated, List

import anndata
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 文件调用

from tool.scgpt_model import ScgptModel

from config import MODEL_PATHS, DICT_PATHS

class CellState(TypedDict):
    file_path: str
    requirement_cls: str
    tissue_name: str
    tissue_id: int

def cell_id2name(tissue_id):
    file_path = DICT_PATHS["cell_id2name"]
    id2name_dict = json.load(open(file_path, 'r'))
    return id2name_dict.get(str(tissue_id), 'Unknown')

def read_adata(file_path):
    return anndata.read(file_path)

def search_cell(state):
    input_data = read_adata(state["file_path"])
    tissue_id = state["tissue_id"]
    tissue_name = state["tissue_name"]

    lora_file = f'{MODEL_PATHS["lora_dir"]}/tissue{tissue_id}/lora_model.pt'
    cell_searcher = ScgptModel(input_data=input_data, data_type="sctab")
    cell_searcher.build_model(model_type="cell", lora_file=lora_file)
    cell_searcher.data_process(data_type="test")
    cell_embeddings, cell_id = cell_searcher.inference(mode="both")
    # TODO: batch inference
    cell_id = cell_id[0] # single inference
    cell_name = cell_id2name(cell_id)

    # cell_checker()

    state_update = {
        "tissue_id": tissue_id,
        "tissue_name": tissue_name,
        "cell_id": cell_id,
        "cell_name": cell_name,
        "cell_embeddings_lora": cell_embeddings,
        "response": "The tissue is " + tissue_name + ", and the cell is " + cell_name + "."
    }

    print(state_update)

    return state_update


if __name__ == "__main__":
    state = CellState(file_path="/data2/lpg_data/test_data/tissue10/1.h5ad", requirement_cls="get_cell", tissue_id=10, tissue_name="bladder")
    result = search_cell(state)
    print(result)
