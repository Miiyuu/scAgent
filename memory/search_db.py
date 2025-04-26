from pymilvus import MilvusClient
import random

import json

from config import DB_PATHS, DICT_PATHS


# 创建客户端连接
client = MilvusClient(DB_PATHS["cell_embedding2"])
# client = MilvusClient("./cell_embedding3.db")


# def query(query_vectors,search_num=10,is_lora=True,search_type="cells",search_data="sctab"):

#     if is_lora:
#         collection_name=f"{search_data}_{search_type}"
#     else:
#         collection_name=f"nolora_{search_data}"
    
#     print(collection_name)

#     nearest_tissue = client.search(
#         collection_name=collection_name,  # 目标集合
#         data=query_vectors,  # 查询向量
#         limit=search_num,  # 返回的实体数量
#         output_fields=["tissue", "cell_type"],  # 指定返回的字段
#     )


#     if search_type=="cells":
#         results=[[(res["entity"]["cell_type"],res["distance"]) for res in result] for result in nearest_tissue]
#     else:
#         results=[[(res["entity"]["tissue"],res["distance"]) for res in result] for result in nearest_tissue]

    
#     return results

def query(query_vectors, search_num=10, is_lora=True, search_type="cells", search_data="sctab", tissue=None):
    """
    查询函数，支持根据 tissue 过滤数据后再检索最近邻向量。

    :param query_vectors: 查询向量列表
    :param search_num: 返回的最近邻数量
    :param is_lora: 是否使用 lora 模型
    :param search_type: 搜索类型，如 "cells" 或 "tissues"
    :param search_data: 数据集名称，如 "sctab"
    :param tissue: 组织名称，用于过滤数据
    :return: 返回最近邻的结果列表
    """
    # 确定集合名称
    if is_lora:
        collection_name = f"{search_data}_{search_type}"
    else:
        collection_name = f"nolora_{search_data}"
    
    print(f"Searching in collection: {collection_name}")

    # 如果指定了 tissue，设置过滤条件
    filter_expr = f"tissue == '{tissue}'" if tissue else None

    # 在 Milvus 中搜索最近邻
    nearest_tissue = client.search(
        collection_name=collection_name,  # 目标集合
        anns_field="vector",  # 向量字段
        data=query_vectors,  # 查询向量
        limit=search_num,  # 返回的实体数量
        output_fields=["tissue", "cell_type"],  # 指定返回的字段
        filter=filter_expr  # 过滤条件
    )

    # 根据搜索类型返回结果
    if search_type == "cells":

        if search_data=="sctab":
            # 1. 读取 JSON 文件
            with open(DICT_PATHS["cell_id2name"], 'r') as file:
                cell_id2name = json.load(file)

            # 2. 反转字典，将名称作为键，编号作为值
            name2cell_id = {v.lower(): k for k, v in cell_id2name.items()}
    
            results = [[(res["distance"],name2cell_id.get(res["entity"]["cell_type"],"none")) for res in result] for result in nearest_tissue]

        else:
            results = [[(res["distance"],res["entity"]["cell_type"]) for res in result] for result in nearest_tissue]
    else:
        results = [[(res["distance"],res["entity"]["tissue"]) for res in result] for result in nearest_tissue]
    
    return results



if __name__ == '__main__':


    # # 创建10000条query_vectors
    # query_vectors = [[random.uniform(-1, 1) for _ in range(512)] for _ in range(10000)]
    # # query_vectors = [[random.random() for _ in range(512)]]
    # res = query_tissue(query_vectors)
    # print(res)



    # 列出所有集合
    collections = client.list_collections()

    # 打印集合名称
    print("Collections in the database:")
    for collection in collections:


        print(collection)

    #     index_info = client.describe_index(collection_name=collection, index_name="flat_index")
    #     print("Index info:", index_info)


    # # # 检查索引是否存在
    # # indexes = client.list_indexes(collection_name="nolora_sctab")
    # # print("Indexes in collection:", indexes)


    # query_vectors = [[random.uniform(-1, 1) for _ in range(512)] for _ in range(1)]


    # res = query(query_vectors,search_num=10,is_lora=False,search_type="cells",search_data="sctab",tissue="liver")

    # print(res)
