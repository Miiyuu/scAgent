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

prompt = ChatPromptTemplate([
    ("system", "You are a helpful Al bot that provides answers based on the user's input and a reference result. Your task is to generate a response that is clear, informative, and helpful. If the result indicates a specific cell type, provide the likely cell type. If the result is ‘unknown’, suggest adding the new cell data to the database. If the result is a confirmation of data update, acknowledge the update and offer further assistance."),
    ("human", "Here are some examples:\n"
              "1. **User Input:**\nThis is the scRNA sequencing data of a cell from human bone marrow tissue. Please help me determine the specific cell type.\n../data/data.h5ad\n"
              "**Result:**\nactivated CD4-positive, alpha-beta T cell\n"
              "**Output:**\nThe cell is likely to be an activated CD4-positive, alpha-beta T cell. Is there anything else that I can help with?"
              "2. **User Input:**\nThis is the scRNA sequencing data of a cell from human bone marrow tissue. Please help me determine the specific cell type.\n../data/data.h5ad\n"
              "**Result:**\nunknown\n"
              "**Output:**\nThe cell is likely to be a novel cell. Would you like to add data about this new cell to our database?"
              "3. User Input:\nYes please! Here is the data file of this novel cell.\n../data/novel_cell.h5ad\n"
              "Result:\nincrement lora model has been saved\n"
              "Output:\nOK! I have added your data to my database! Is there anything else that I can help with?"),
    ("human", "Now, based on the given user input and reference result, generate a response that is clear, informative, and helpful. Note that you can imitate the example, but do not generate exactly the same response.\n"
              "**User Input:**\n{user_input}\n"
              "**Result:**\n{result}"
              "**Output:**\n"),
])



runnable = prompt | llm
def output_generator(state):
    # print(state)
    response = (runnable.invoke({"user_input": state["user_input"], "result": state["response"]}))
    # print(response.content)

    return {"response": response.content}


if __name__ == "__main__":
    result = output_generator(
        {
            "user_input": "/data2/lpg_data/test_data/ts_liver/1.h5ad\n This is the scRNA sequencing data of a cell. Please help me determine the specific cell type.",
            "response": "CD4+ T Cell"
        }
    )
    print(result)
