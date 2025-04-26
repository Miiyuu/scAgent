from scAgent.memory.search_db import query



def novel_assess(state):
    cell_embeddings = state["cell_embeddings_lora"]
    # tissue_name = state["tissue_name"]

    # 设定距离阈值
    DISTANCE_THRESHOLD = 10  # 这个阈值值需要根据实际情况调整

    # 只查询最近的1个细胞
    results = query(cell_embeddings, search_num=1, is_lora=True, search_type="cells", search_data="sctab")  

    print(results)

    # 获取最近细胞的距离（适用single cell）
    nearest_distance = results[0][0][0]  # 假设返回的结果包含距离信息

    # 根据距离判断是否需要新颖性检测
    if nearest_distance > DISTANCE_THRESHOLD:
        state["need_novel_detect"] = True
    else:
        state["need_novel_detect"] = False

    return state