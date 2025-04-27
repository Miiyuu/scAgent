import functools
import operator
import random
from typing import Annotated, List, Literal

import numpy as np
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.types import Send
from pandas import array
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent.input_analyzer import input_analyzer
from agent.tissue_searcher import search_tissue
from agent.cell_searcher import search_cell
from agent.output_generator import output_generator
from agent.ft_increment import increment_cell
from agent.novel_assess import novel_assess 
from agent.novel_detection.openset_identify import novel_detect

class State(TypedDict):
    # messages: Annotated[list, add_messages]
    user_input: str
    file_path: str
    requirement: str
    tissue_id: int
    tissue_name: str
    cell_id: int
    cell_name: str
    response: str
    cell_embeddings_lora: np.ndarray
    need_novel_detect: bool

def route_from_analyzer(state: State):
    # 修改这里：使用字典访问方式
    if state["requirement"] != "add_cell":  # 注意这里使用 requirement_cls
        return "tissue_searcher"  # 当不是add_cell时，去搜索组织
    return "increment_cell"  # 当是add_cell时，直接搜索细胞

def route_to_novel_detect(state: State):
    if state["need_novel_detect"]:
        return "novel_detect"
    return "output_generator"


def init_graph1():
    graph = StateGraph(State)

    graph.add_node("input_analyzer", input_analyzer)
    graph.add_node("tissue_searcher", search_tissue)
    graph.add_node("cell_searcher", search_cell)
    graph.add_node("output_generator", output_generator)

    graph.add_edge("input_analyzer", "tissue_searcher")
    graph.add_edge("tissue_searcher", "cell_searcher")
    graph.add_edge("cell_searcher", "output_generator")
    graph.add_edge("output_generator", END)

    graph.set_entry_point("input_analyzer")

    return graph.compile()

def init_graph():
    graph = StateGraph(State)

    graph.add_node("input_analyzer", input_analyzer)
    graph.add_node("tissue_searcher", search_tissue)
    graph.add_node("cell_searcher", search_cell)
    graph.add_node("increment_cell", increment_cell)
    graph.add_node("novel_assess", novel_assess)
    graph.add_node("novel_detect", novel_detect)
    graph.add_node("output_generator", output_generator)

    # 使用条件路由函数添加条件边
    graph.add_conditional_edges(
        "input_analyzer", 
        route_from_analyzer,
        {
            "tissue_searcher": "tissue_searcher",
            "increment_cell": "increment_cell",
        }
    )

    graph.add_edge("tissue_searcher", "cell_searcher")

    graph.add_edge("cell_searcher", "novel_assess")

    graph.add_conditional_edges(
        "novel_assess",
        route_to_novel_detect,
        {
            "novel_detect": "novel_detect",
            "output_generator": "output_generator"
        }
    )

    graph.add_edge("novel_detect", "output_generator")
    graph.add_edge("increment_cell", "output_generator")
    # graph.add_edge("output_generator", "input_analyzer")  # 添加从output返回到input的边,形成闭环
    graph.add_edge("output_generator", END)
    
    graph.set_entry_point("input_analyzer")

    return graph.compile()

def draw_graph(graph, save_path):
    try:
        # display(Image(graph.get_graph().draw_mermaid_png()))
        # Save graph
        with open(save_path, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print("Graph saved to output.png")
    except Exception as e:
        # This requires some extra dependencies and is optional
        print("Can't show graph.")
        print(f"错误信息: {str(e)}")

if __name__ == '__main__':
    graph = init_graph()
    draw_graph(graph, "output.png")

