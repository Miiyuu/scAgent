# python3
# Please install OpenAI SDK first：`pip3 install openai`
import json
import os
import re
import time

from openai import OpenAI, OpenAIError
import openai

# TODO: check proxy
# openai.proxy = {
#     'http':'http://10.82.210.138:7890',
#     'https':'http://10.82.210.138:7890'
# }

# os.environ["HTTP_PROXY"] ="http://10.82.210.138:7890"
# os.environ["HTTPS_PROXY"] ="http://10.82.210.138:7890"

# os.environ["HTTP_PROXY"] ="http://10.87.5.131:7890"
# os.environ["HTTPS_PROXY"] ="http://10.87.5.131:7890"

class DeepSeek:
    def __init__(self):
        super().__init__()
        # self.client = OpenAI(api_key="sk-adbb3d5df8d244e280a530f8ee99af9e", base_url="https://api.deepseek.com")
        self.client = OpenAI(api_key="sk-44a071209dad4d2fb2e21827c773e02e", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def model_generate(self, prompt, max_retries=20, retry_delay=2):
        retries = 0
        while retries < max_retries:
            try:
                # 尝试发送请求
                response = self.client.chat.completions.create(
                    # model="deepseek-chat",
                    model="qwen-plus",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    timeout=20  # 设置超时时间，避免请求卡住
                )
                res = response.choices[0].message.content

                # 尝试提取并解析 JSON
                match = re.search(r'\{.*?\}', res, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        res = json.loads(json_str)
                        return res  # 如果成功，直接返回结果
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        return None  # 返回 None 表示 JSON 解析失败
                else:
                    print("No JSON found in the text.")
                    return None  # 返回 None 表示未找到 JSON

            except (OpenAIError, Exception) as e:  # 捕获网络请求或其他异常
                print(f"Attempt {retries + 1} failed: {e}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)  # 等待一段时间后重试
                else:
                    print("Max retries reached. Giving up.")
                    return None  # 返回 None 表示请求失败

        return None  # 默认返回 None

    # 处理大模型的输出
    def process_output(self, output):
        try:
            # 如果 output 是字符串，尝试解析为 JSON
            if isinstance(output, str):
                result = json.loads(output)
            # 如果 output 已经是字典，直接使用
            elif isinstance(output, dict):
                result = output
            else:
                return "error", "Invalid output type: expected str or dict"

            # 检查必要字段
            if 'is_new_cell' not in result or 'explanation' not in result:
                return "error", "Invalid output format: missing required keys"

            # 根据 is_new_cell 的值返回结果
            if result['is_new_cell'] == 'yes':
                return "unknown", result['explanation']
            else:
                if 'cell_type' not in result:
                    return "error", "Invalid output format: missing 'cell_type' key"
                return result['cell_type'], result['explanation']

        except json.JSONDecodeError:
            return "error", "Invalid JSON format"
        except Exception as e:
            return "error", f"An unexpected error occurred: {str(e)}"



        
def build_prompt1(distances_lora,distances_nolora):
    prompt=f"""
    You are a biologist skilled in identifying cell types. There are two cell feature extractors: the first one has been pre-trained using self-supervised learning on a large amount of single-cell data, and the second one has been fine-tuned based on the first, which may lead to feature collapse and loss of some cell features. There are two vector databases composed of vectors extracted by these two feature extractors, respectively. Now, a single-cell data is processed by both feature extractors and searched in the corresponding vector databases. Please determine whether this cell is a new cell based on the search results. If not, provide the specific cell type. Generally, if the average distance retrieved from the first vector database is greater than 5, the cell should be considered a new cell. Otherwise, further judgment should be made based on the results retrieved from the second vector database. The output should be in JSON format: {{is_new_cell (yes or no), cell_type, explanation}}.
    ## Input
    ## Search results
    ### First vector database:
    #### Distance to the nearest cell cluster center: {distances_nolora[0]}
    #### Most frequent cell type among the nearest 10 cells: {distances_nolora[1]}
    #### Average distance of the nearest 10 cells: {distances_nolora[2]}
    ### Second vector database:
    #### Distance to various cell cluster centers: (cell type, distance)
    {distances_lora[0]}
    #### Nearest 10 cells: (distance, cell type)
    {distances_lora[1]}
    Please make a judgment based on the input.
    """

    # print(prompt)

    return prompt

def build_prompt(distances_lora, distances_nolora):
    # 反转 distances_lora[1] 为 (cell, distance) 格式
    # nearest_cells_lora = [(cell, round(distance, 3)) for distance, cell in distances_lora[1]]
    nearest_cells_lora = [(cell, round(distance, 3)) for distance, cell in distances_lora[1]]

    prompt = f"""## **Instruction**

You are a biologist skilled in identifying cell types. There are two vector databases:
1. **Database 1 (db1)**: Features from a self-supervised model trained on extensive single-cell data
2. **Database 2 (db2)**: Features fine-tuned from db1's model (may exhibit feature collapse)
Now, a single-cell data is processed by both feature extractors and searched in the corresponding vector databases. Based on the search results, you need to determine whether this cell is a new cell. If it is not a new cell, provide the specific cell type.  

### **Output Format**  
The output should be a **JSON object** with the following keys:  
- `"is_new_cell"`: `"yes"` or `"no"`.  
- `"cell_type"`: `"unknown"` for new cells or the identified cell type for existing cells.  
- `"explanation"`: A clear description of the decision.  

---

## **Few-Shot Examples**

### **Few-Shot 1: Cell is a new cell (High average distance in `db1`)**

#### Input:  
**Search results**  
- **First vector database:**  
  - Most frequent cell type among the nearest 10 cells: "T-cell"  
  - Distance to the nearest cell cluster center: 30.123  
  - Average distance of the nearest 10 cells: 27.876  

- **Second vector database:**  
  - Distance to various cell cluster centers: [(“T-cell”, 126.245), (“B-cell”, 236.412)]  
  - Nearest 10 cells: [("T-cell", 326.245), ("T-cell", 327.412), ("T-cell", 326.245), ("T-cell", 326.245), ("T-cell", 326.245), ("T-cell", 326.245), ("T-cell", 327.412), ("T-cell", 326.245), ("T-cell", 326.245), ("T-cell", 326.245)]  

#### Output:  
```json
{{
  "is_new_cell": "yes",
  "cell_type": "unknown",
  "explanation": "db1's high average distance (27.876 >25 threshold) strongly indicates a new type. While db2 shows T-cell dominance, its magnified distances (feature collapse) cannot override db1's clear evidence."
}}
```

------

### **Few-Shot 2: Cell is a new cell (High similarity and variety in `db2`)**

#### Input:

**Search results**

- **First vector database:**
  - Most frequent cell type among the nearest 10 cells: "T-cell"
  - Distance to the nearest cell cluster center: 14.932
  - Average distance of the nearest 10 cells: 14.765
- **Second vector database:**
  - Distance to various cell cluster centers: [(“T-cell”, 26.123), (“B-cell”, 26.145), ("macro phage", 26.998), ("Bergmann glial cell", 27.783)]
  - Nearest 10 cells: [("T-cell", 26.120), ("T-cell", 26.124), ("B-cell", 27.145), ("macro phage", 27.548), ("Bergmann glial cell", 27.551),("macro phage", 27.558),("macro phage", 27.598),("B-cell", 27.602),("Bergmann glial cell", 27.613),("Bergmann glial cell", 27.644)]

#### Output:

```json
{{
  "is_new_cell": "yes",
  "cell_type": "unknown",
  "explanation": "db1's borderline distance (14.765) requires db2 confirmation. db2 shows: 1) All cluster distances >25 (equivalent to >15 in db1), 2) Diverse neighbors suggesting no clear type match. Consensus: new cell."
}}
```

------

### **Few-Shot 3: Cell is not new (Matches existing cell type in `db2`)**

#### Input:

**Search results**

- **First vector database:**
  - Most frequent cell type among the nearest 10 cells: "T-cell"
  - Distance to the nearest cell cluster center: 1.951
  - Average distance of the nearest 10 cells: 1.845
- **Second vector database:**
  - Distance to various cell cluster centers: [(“T-cell”, 1.785), (“B-cell”, 4.032)]
  - Nearest 10 cells: [("T-cell", 3.785), ("T-cell", 3.812), ("T-cell", 3.945),("T-cell", 3.985), ("T-cell", 3.997), ("T-cell", 3.998),("T-cell", 4.012), ("T-cell", 4.123), ("T-cell", 4.156), ("B-cell", 4.163)]

#### Output:

```json
{{
  "is_new_cell": "no",
  "cell_type": "T-cell",
  "explanation": "db1 shows strong proximity (1.845 <15). db2 confirms: 1) Clear T-cell cluster proximity (relative distance 1.785 vs 4.032), 2) Neighbor consistency (90% T-cells). Confident match."
}}
```

------

### **Few-Shot 4: Cell is not new (Matches existing cell type in `db2`)**

#### Input:

**Search results**

- **First vector database:**
  - Most frequent cell type among the nearest 10 cells: "Monocyte"
  - Distance to the nearest cell cluster center: 1.982
  - Average distance of the nearest 10 cells: 1.785
- **Second vector database:**
  - Distance to various cell cluster centers: [("Monocyte", 1.876), ("Granulocyte", 4.654)]
  - Nearest 10 cells: [("Monocyte", 1.722), ("Monocyte", 1.757), ("Monocyte", 1.799), ("Monocyte", 1.826), ("Monocyte", 1.828), ("Monocyte", 1.830)("Monocyte", 1.842), ("Monocyte", 1.858), ("Monocyte", 1.912), ("Monocyte", 1.922)]

#### Output:

```json
{{
  "is_new_cell": "no",
  "cell_type": "Monocyte",
  "explanation": "Clear consensus: 1) db1 distance (1.785) far below 15 threshold, 2) db2 shows tight cluster of Monocyte (all neighbors same type with consistent relative distances)."
}}
```

    Now, based on the search results, you need to determine whether this cell is a new cell. If it is not a new cell, provide the specific cell type:

    ## Input
    ## Search results
    ### First vector database:
    #### Most frequent cell type among the nearest 10 cells: {distances_nolora[1]}
    #### Distance to the nearest cell cluster center: {distances_nolora[0]}
    #### Average distance of the nearest 10 cells: {round(distances_nolora[2], 3)}

    ### Second vector database:
    #### Distance to various cell cluster centers: (cell type, distance)
    {[(cell_type, round(distance, 3)) for cell_type, distance in distances_lora[0]]}
    #### Nearest 10 cells: (cell type, distance)
    {nearest_cells_lora}

        """

    return prompt
