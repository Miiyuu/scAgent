import sys
import re
import os
# 将项目的根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import scanpy as sc

import pickle

from tool.scgpt_model import ScgptModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from types import SimpleNamespace
import copy
import pandas as pd
import time
import warnings
import pandas as pd
import pickle
import torch
from anndata import AnnData
import scanpy as sc
# import scvi
import seaborn as sns
import numpy as np
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

import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# # 设置随机种子
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# # PyTorch GPU
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
# sc._settings.ScanpyConfig.seed = seed


sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
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

from tqdm import tqdm

from config import DICT_PATHS

import json

# TensorFlow

config = dict(
    lr=1e-2,
    amp=True,  # Automatic Mixed Precision
    schedule_interval = 1,
    schedule_ratio=0.9
)

epochs=10

# 将字典转换为对象
config = SimpleNamespace(**config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = scg.logger

from config import MODEL_PATHS, DICT_PATHS, DATA_PATHS


class increment_Trainer:
    def __init__(self, tissue_id):
        
        lora_dir=MODEL_PATHS["lora_dir"]

        lora_file=lora_dir+"/"+tissue_id+"/lora_model.pt"

        self.scgptmodel = ScgptModel(data_type="sctab")

        self.scgptmodel.build_model(model_type="cell",lora_file=lora_file)

        self.vocab = self.scgptmodel.parameter["vocab"]

        model=self.scgptmodel.model



        for n, p in model.named_parameters():

            if 'lora_' not in n and 'bn' not in n:
                p.requires_grad = False
            # else:
            #     print(n)

            # if 'lora_route' not in n and 'bn' not in n:
            #     p.requires_grad = False
            # else:
            #     print(n)

        
        
        self.criterion = masked_mse_loss
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_dab = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
        # )
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-4 if config.amp else 1e-8
        )



        # 统计优化参数的总量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, config.schedule_interval, gamma=config.schedule_ratio
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

        self.model=model
    
    def process_data(self,data,data_type):

        scgptmodel=self.scgptmodel

        scgptmodel.input_data=data

        scgptmodel.data_process(data_type=data_type,drop_last=True)

        return copy.deepcopy(scgptmodel.data_loader)
        # return scgptmodel.data_loader


    def train(self,model: nn.Module, loader: DataLoader,epoch) -> None:
        """
        Train the model for one epoch.
        """
        model.train()


        (
            total_loss,
            total_cls,
        ) = (0.0, 0.0)
        total_error = 0.0
        start_time = time.time()

        num_batches = len(loader)

        ## 4. 打印模型权重
        # print(f'Epoch {epoch} - Model weights:')
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f'{name}: {param.data}')
        # # 获取模型的 state_dict
        # model_state_dict = model.state_dict()




        for batch, batch_data in enumerate(tqdm(loader, desc="Training")):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                    # generative_training=False
                )

                loss = 0.0
                metrics_to_log = {}

                output_values=output_dict["cls_output"]

    

                preds = output_values.argmax(1).cpu().numpy()
        

                loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)

                # if batch==0:
                #     # 设置输出精度为小数点后 10 位
                #     torch.set_printoptions(precision=20)

                #     print(output_values)

                #     print(loss_cls)


                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)


            model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            CLS=True



            # # 4. 打印模型权重
            # print(f'Epoch {epoch} - Model weights:')
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'{name}: {param.data}')


            # # 在backward之后、clip之前打印梯度指纹
            # grad_signature = hash(tuple(p.grad.sum().item() for p in model.parameters()))
            # print(f"Gradient Hash: {grad_signature}")

            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_cls += loss_cls.item()

            log_interval=3
            total_error += error_rate
            if batch % log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_cls = total_cls / log_interval
   
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"cls {cur_cls:5.2f} | " if CLS else "")
                    + (f"err {cur_error:5.2f} | " if CLS else "")
    
                )
                total_loss = 0
                total_cls = 0
                total_error = 0
                start_time = time.time()

    def evaluate(self,model: nn.Module, loader: DataLoader,return_raw: bool = False) -> float:

        model.eval()

        total_loss = 0.0
        total_error = 0.0
        total_num = 0
        predictions = []    
        true_labels=[]
        with torch.no_grad():
            CLS=True
            #Corn

            for batch_data in tqdm(loader,desc="testing"):
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)
                true_labels.extend(celltype_labels.cpu().numpy().tolist())
                src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=None,
                        CLS=CLS,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=False,
                        # generative_training = False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = self.criterion_cls(output_values, celltype_labels)


                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)

        if return_raw:
            return np.concatenate(predictions, axis=0),true_labels

        return total_loss / total_num, total_error / total_num

    def test(self,test_data):
        
        if self.best_model is None:
            raise ValueError("The best model is not set. Please train the model first.")
        
        model=self.best_model
        
        model.eval()

        test_loader = self.process_data(test_data,data_type="test")

        predictions,true_labels = self.evaluate(
            model,
            loader=test_loader,
            return_raw=True,
        )

        # print(predictions)

        # celltypes_labels=test_data.obs["celltype"].tolist()

        celltypes_labels=true_labels


        # compute accuracy, precision, recall, f1
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(celltypes_labels, predictions)
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}"
        )

    def fine_tuning(self,train_data,val_data):

        best_val_loss = float("inf")
        best_avg_bio = 0.0
        best_model = None

        self.train_loader=self.process_data(train_data,data_type="train")
        self.valid_loader=self.process_data(val_data,data_type="val")

        model=self.model

        val_loss_list=[]

        for epoch in range(1, epochs + 1):
  
            epoch_start_time = time.time()

            self.train(
                model,
                loader=self.train_loader,
                epoch=epoch,
            )

            val_loss, val_err = self.evaluate(
                model,
                loader=self.valid_loader
            )

            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
            )
            logger.info("-" * 89)

            val_loss_list.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                # best_model_epoch = epoch
                logger.info(f"Best model with score {best_val_loss:5.4f}")


            self.scheduler.step()

    
        self.best_model = best_model

        self.val_loss_list=val_loss_list  


def type2id(adata,isnew=False):

    unique_cell_types_order = pd.read_csv(DICT_PATHS["unique_cell_types_order"])['cell_type_id'].tolist()

    # unique_cell_types_order排序
    unique_cell_types_order = sorted(unique_cell_types_order)

    if isnew:

        # 找到当前列表中最大的数字
        max_id = max(unique_cell_types_order)
        # 添加一个新的类别，编号为最大数字加 1
        new_category_id = max_id + 1
        unique_cell_types_order.append(new_category_id)


    # # 创建一个包含排序后类别的 Categorical 类型
    sorted_categories = pd.Categorical(adata.obs["cell_type"], categories=unique_cell_types_order)

    # 生成类别编码
    celltype_id_labels = sorted_categories.codes

    return celltype_id_labels




def make_train_data(tissue_id,new_adata,size,new_size):

    data_dir = Path(f"{DATA_PATHS['sctab_data']}/{tissue_id}/")
    adata_train = sc.read(data_dir / "train.h5ad")

    adata_labels = adata_train.obs["cell_type"]

    # 获取adata中每类细胞的标签
    unique_labels = np.unique(adata_labels)
    for i,label in enumerate(unique_labels):
        indices = np.where(adata_labels == label)[0]
        if len(indices) > size:
            selected_indices = np.random.choice(indices, size=size, replace=False)
        else:
            selected_indices = indices
        
        if i ==0:
            selected_adata=adata_train[selected_indices]
        else:
            selected_adata=selected_adata.concatenate(adata_train[selected_indices])
    
    # print(selected_adata.obs["cell_type"].tolist())

    # 从 new_adata 中随机选取 new_size 个样本
    new_adata_indices = np.random.choice(new_adata.n_obs, size=new_size, replace=False)
    new_adata_selected = new_adata[new_adata_indices]

    # 将选中的 new_adata 数据与 selected_adata 进行融合
    combined_adata = selected_adata.concatenate(new_adata_selected)

    # # 剩余的 new_adata 数据作为测试集
    # remaining_indices = np.setdiff1d(np.arange(new_adata.n_obs), new_adata_indices)
    # test_adata = new_adata[remaining_indices]


    return combined_adata


def increment_cell(state):


    seed_everything(40)

    tissue_name = state["tissue_name"]

    tissue_id2name = json.load(open(DICT_PATHS["tissue_id2name"]))
    tissue_name2id = {v: k for k, v in tissue_id2name.items()}
    tissue_id = f"tissue{tissue_name2id[tissue_name]}"

    new_data_path=state["file_path"] 


    new_adata=sc.read(new_data_path)

    size=new_adata.n_obs


    combined_adata=make_train_data(tissue_id,new_adata,size,new_size=size)

    combined_adata.obs["celltype"]=type2id(combined_adata,isnew=True)

    indices = np.arange(combined_adata.n_obs)
    
    # 使用 train_test_split 随机拆分索引
    train_indices, val_indices = train_test_split(
        indices, 
        train_size=0.9, 
        # random_state=42,  # 设置随机种子以确保结果可复现
        stratify=combined_adata.obs["celltype"]  # 按细胞类型分层抽样
    )

    # 根据索引拆分 AnnData 对象
    train_adata = combined_adata[train_indices]
    val_adata = combined_adata[val_indices]

    increment_trainer=increment_Trainer(tissue_id)

    increment_trainer.fine_tuning(train_adata,val_adata)

    save_dir=f"{MODEL_PATHS['increment_lora_dir']}"
    
    # 检查 save_dir 是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_model=increment_trainer.best_model

    torch.save(lora.lora_state_dict(best_model),f"{save_dir}/{tissue_name}_lora.pt")

    state_update = {
        "response": "increment lora model has been saved"
    }

    return state_update



