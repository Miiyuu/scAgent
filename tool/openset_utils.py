import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,random_split
import h5py
import numpy as np
from tqdm import tqdm
import copy
import pickle
import random
from collections import Counter
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpenClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(OpenClassifierModel, self).__init__()
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 创建嵌入层，用于生成 CLS Token 的嵌入向量
        self.cls_token_embedding = nn.Embedding(1, input_dim)

        # MLP Layer
        self.mlp = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2)  # Output layer with 2 units for binary classification
        )
        
    def forward(self, x):
        # Pass through Transformer layer

        x = self.transformer_encoder(x)

        # 通过 MLP 层
        output = self.mlp(x)
        
        return output

def build_openset_model(input_dim, hidden_dim,num_heads,num_layers,model_path):
    model = OpenClassifierModel(input_dim, hidden_dim, num_heads, num_layers).to(device)
    # 加载最佳模型权重
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def predict_mutil_model(data,emblist_mean,model_dict):


    values={}
    predicteds={}

    for category,mean_point in emblist_mean.items():

        model=model_dict[category]

        input=np.concatenate((mean_point,data))


        input=torch.tensor(input).to(device).unsqueeze(0).float()   
        #增加一个维度，并转为float类型
    
        output=model(input)

        output=torch.softmax(output,dim=1)

        value, predicted = torch.max(output.data, 1)
        values[category]=value.item()
        predicteds[category]=predicted.item()


    return values,predicteds

    
    



    



