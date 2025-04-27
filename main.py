import argparse
import io

import matplotlib.pyplot as plt
from PIL import Image

from graph import init_graph, init_graph1
from contextlib import redirect_stdout, redirect_stderr
import os
import scanpy as sc

from config import DATA_PATHS

def show_graph(graph):
    image = graph.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(image))

    plt.imshow(image)
    plt.axis("off")
    plt.show()

def run(user_input: str):
    graph = init_graph()
    # show_graph(graph)
    for event in graph.stream({"user_input": user_input}, stream_mode="values"):
        print("\n", "-" * 50, "-" * 50)
        # print(event)
    response = event["response"]
    print(response)
    return

def run1(user_input: str):
    # 只有细胞类型识别的图
    graph = init_graph1()
    # show_graph(graph)
    for event in graph.stream({"user_input": user_input}, stream_mode="values"):
        print("\n", "-" * 50, "-" * 50)
        # print(event)
    response = event["response"]
    print(response)
    return


if __name__ == "__main__":

    # cancer_path="/data/lpg/codes/scAgent/share/agent/openClassifier/cancer_data/abnormal/liver.h5ad"
    # cancer_data = sc.read_h5ad(cancer_path)
    # cancer_data=cancer_data[0]
    
    # # 保存cancer_data到指定目录
    # save_path = "/data/lpg/codes/scAgent/miyu/agent/test/cancer_data.h5ad"
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # cancer_data.write_h5ad(save_path)


    # test_path=DATA_PATHS["test_data"]
    # test_path="/data/lpg/codes/scAgent/miyu/agent/test/cancer_data.h5ad"
    test_path=DATA_PATHS["test_increment_adata"]

    
    # user_input = f"{test_path}\n This is the scRNA sequencing data of a cell. Please help me determine the specific cell type."
    

    user_input = f"{test_path}\n This is the scRNA sequencing data of a new cell type from liver tissue. Please help me add this cell type to the model."
    run(user_input)
    # run1(user_input)
