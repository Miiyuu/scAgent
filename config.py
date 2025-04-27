"""
配置文件，包含所有在代码中使用的路径
"""

# 基础路径
BASE_DIR = "/data/lpg/codes/scAgent/miyu/agent/scAgent"
MODEL_DIR = f"{BASE_DIR}/model"
DICT_DIR = f"{BASE_DIR}/dict"
DATA_DIR = f"/data2/lpg_data"


# 数据路径
DATA_PATHS = {
    "test_data": f"{BASE_DIR}/test/tissue15_1.h5ad",
    "test_cancer_data": f"{BASE_DIR}/test/cancer_data.h5ad",
    "test_increment_adata": f"{BASE_DIR}/test/new_adata.h5ad",
    "sctab_data": f"{DATA_DIR}/sctab_data"
}


# 模型路径
MODEL_PATHS = {
    "pretrained": f"{MODEL_DIR}/pretrained_models",
    "tissue_lora_model": f"{MODEL_DIR}/tissue_lora.pt",
    "tissue_lora_model_ts": f"{MODEL_DIR}/tissue_lora_ts.pt",
    "cls_file": f"{MODEL_DIR}/cls_params_200.pth",
    "lora_dir": f"{DATA_DIR}/sctab_scgpt_lora_wo_normed",
    "increment_lora_dir": f"{BASE_DIR}/memory/increment_lora"
}


DICT_PATHS = {
    "id2type": f"{DICT_DIR}/id2type.json", # 细胞类型预测id到真实id的映射
    "cell_id2name": f"{DICT_DIR}/cell_id2name.json", # 细胞类型真实id到名称的映射
    "tissue_id2name": f"{DICT_DIR}/tissue_id2name.json", # 组织类型id到名称的映射
    "tissue_general_mapping": f"{DICT_DIR}/tissue_general_mapping.json", # 组织类型预测id到真实id的映射
    "unique_cell_types_order": f"{DICT_DIR}/unique_cell_types.csv", # 细胞类型id顺序
    "center_lora": f"{DICT_DIR}/center_lora.pkl", 
    "center_nolora": f"{DICT_DIR}/center_nolora.pkl" 
}


# 数据库路径
DB_PATHS = {
    "cell_embedding": f"{BASE_DIR}/memory/cell_embedding.db",
    "cell_embedding2": f"{BASE_DIR}/memory/cell_embedding2.db"
}

