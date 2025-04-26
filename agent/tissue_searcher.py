import json
import operator
import os
import random
import re
import sys
from typing import Annotated, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 文件调用

# from tool.scgpt import ScgptModel
from tool.scgpt_model import ScgptModel
import anndata

from config import DICT_PATHS



class TissueState(TypedDict):
    file_path: str
    requirement_cls: str

def tissue_id2name(tissue_id):
    file_path = DICT_PATHS["tissue_id2name"]
    id2name_dict = json.load(open(file_path, 'r'))
    return id2name_dict.get(str(tissue_id), 'Unknown')

def tissue_general_mapping(tissue_id):
    file_path = DICT_PATHS["tissue_general_mapping"]
    mapping_dict = json.load(open(file_path, 'r'))
    # 以mapping_dict的value为键，key为值，构建新的字典
    new_dict = {v: k for k, v in mapping_dict.items()}
    return new_dict.get(tissue_id, 'Unknown')

def read_adata(file_path):
    return anndata.read(file_path)

def search_tissue(state):
    input_data = read_adata(state["file_path"])
    searcher = ScgptModel(input_data=input_data,data_type="sctab")

    searcher.build_model(model_type="tissue")
    searcher.data_process(data_type="test")

    tissue_id = searcher.inference()[0] # single inference
    tissue_id = tissue_general_mapping(tissue_id)
    tissue_name = tissue_id2name(tissue_id)

    state_update = {
        "tissue_id": tissue_id,
        "tissue_name": tissue_name,
        "response": "The tissue is " + tissue_name
    }
    return state_update


if __name__ == "__main__":
    state = TissueState(file_path="/data2/lpg_data/test_data/tissue10/1.h5ad", requirement_cls="get_organ")
    result = search_tissue(state)
    print(result)
