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

api_key = os.getenv("DEEPSEEK_API_KEY")
# os.environ["HTTP_PROXY"] ="http://10.87.5.131:7890"
# os.environ["HTTPS_PROXY"] ="http://10.87.5.131:7890"

# llm = ChatOpenAI(api_key=api_key, model="deepseek-chat", base_url="https://api.deepseek.com")
llm = ChatOpenAI(api_key=api_key, model="deepseek-v3", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# prompt = ChatPromptTemplate([
#     ("system", "You are a helpful AI bot that classifies user requests into categories and returns the result in JSON format."),
#     ("human", "Here are the classification rules:"
#               "1. The user wants to obtain the cell type and he has not provided any information of the tissue."
#               "Output: {{'requirement': 'get_cell', 'tissue': 'cell's tissue', 'file_path': 'your_file_path'}}"
#               "2. The user wants to obtain the tissue type, and he has provided tissue name."
#               "Output: {{'requirement': 'get_cell_w_tissue', 'tissue': null, 'file_path': 'your_file_path'}}"
#               "3. The user wants to obtain the tissue type."
#               "Output: {{'requirement': 'get_tissue', 'tissue': null, 'file_path': 'your_file_path'}}"
#               "4. The user wants to add a new cell and has provided the tissue to which the cell belongs."
#               "Output: {{'requirement': 'add_cell', 'tissue': 'cell's tissue', 'file_path': 'your_file_path'}}"),
#     ("human", "Here is an example:"
#               "User Request: This is the scRNA sequencing data of a cell from human bone marrow tissue. Please help me determine the specific cell type.\n ../data/lung/to_infer.h5ad"
#               "Output: {{'requirement_cls': 'query', 'requirement': 'get_cell_w_tissue', 'tissue': 'bone_marrow', 'file_path': '../data/lung/to_infer.h5ad'}}"),
#     ("human", "Now, based on the given user requirement, classify the requirement and only output the JSON object. Don't forget the 'file_path'!"),
#     ("human","User Request: {user_input}""Output: ")
# ])

prompt = ChatPromptTemplate([
    ("system", "You are a helpful AI bot that classifies user requests into categories and returns the result in JSON format."),
    ("human", "Here are the classification rules:\n"
              "1. If the user wants to obtain the **cell type** and has **not provided any information of the tissue**, classify as:\n"
              "   Output: {{\"requirement\": \"get_cell\", \"tissue\": null, \"file_path\": \"your_file_path\"}}\n\n"
              "2. If the user wants to obtain the **cell type** and has **provided the tissue name**, classify as:\n"
              "   Output: {{\"requirement\": \"get_cell_w_tissue\", \"tissue\": \"tissue name\", \"file_path\": \"your_file_path\"}}\n\n"
              "3. If the user wants to obtain the **tissue type**, classify as:\n"
              "   Output: {{\"requirement\": \"get_tissue\", \"tissue\": null, \"file_path\": \"your_file_path\"}}\n\n"
              "4. If the user wants to **add a new cell** and has provided the tissue to which the cell belongs, classify as:\n"
              "   Output: {{\"requirement\": \"add_cell\", \"tissue\": \"tissue name\", \"file_path\": \"your_file_path\"}}\n"),
    ("human", "Here is an example:\n"
              "User Request: This is the scRNA sequencing data of a cell from human bone marrow tissue. Please help me determine the specific cell type.\n"
              "../data/to_infer/1.h5ad\n\n"
              "Output: {{\"requirement\": \"get_cell_w_tissue\", \"tissue\": \"bone_marrow\", \"file_path\": \"../data/lung/to_infer.h5ad\"}}"),
    ("human", "Now, based on the given user requirement, classify the requirement and only output the JSON object. Don't forget the 'file_path'!"),
    ("human", "User Request: {user_input}\nOutput: ")
])



runnable = prompt | llm
def input_analyzer(state):
    print(state)
    response = (runnable.invoke({"user_input": state["user_input"]}))
    # print(response)

    json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in the input string")
    json_string = json_match.group(0)
    json_string = json_string.strip("```json").strip("```").strip()
    json_string = json_string.replace("\n", "").replace(" ", "")
    res_json = json.loads(json_string)
    # 如果没有"file_path"键，则报错
    assert "file_path" in res_json, "No 'file_path' key found in the JSON object"
    file_path = res_json["file_path"]
    requirement = res_json["requirement"]
    tissue_name = res_json.get("tissue", None)

    return {"file_path": file_path, "requirement": requirement, "tissue_name": tissue_name}


if __name__ == "__main__":
    result = input_analyzer(
        {
            "user_input": "/data2/lpg_data/test_data/ts_liver/1.h5ad\n This is the scRNA sequencing data of a cell. Please help me determine the specific cell type."
        }
    )
    print(result)
