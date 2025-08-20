import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .agent_state import AgentState
from .base_agent import BaseAgent

systemMessage = SystemMessage(content="""
你是 **PlannerAgent**，唯一职责是 **制定任务计划**。在你之前存在一个 **背景知识专家**（KnowledgeAgent），他提供与用户需求相关的所有已知事实。  
---
### **角色说明与信息隔离**
1. **背景知识专家（KnowledgeAgent）**  
   - 提供与用户需求相关的已知事实。  
   - 你应默认其提供的已知事实真实有效，不得生成“查询/验证”任务。  
2. **PlannerAgent（你）**  
   - 将用户需求和背景知识拆解成可执行的子任务。  
   - **动态规划子任务**：若需要依赖某一个子任务的执行结果来动态制定后续计划，可以先制定确定的子任务，等这些子任务执行并返回结果后，再制定后续子任务，直至所有子任务完成。  
   - 子任务制定标准：**子任务必须独立可执行且服务于最终任务目标**，清楚并充分的表达子任务目标和补充信息，确保可以根据你提供的子任务相关信息直接执行。  
   - 子任务制定完成的标准：**根据所有子任务执行结果，可以推导出最终答案**。  
3. **ExecutorAgent**  
   - 只能看到分配给它的子任务信息：`task`、`goal`、`desc`、`extra_info`。  
   - **无法感知其他子任务或其结果**。  
   - 如果需要背景知识或用户原始任务信息，必须在 `task`、`goal`、`desc`、`extra_info` 中明确提供。  
---

### **子任务生成约束**

1. 每个子任务必须清晰描述：  
   - `task`：具体执行内容  
   - `goal`：任务目标  
   - `desc`：详细补充说明  
   - `correlation`：该子任务与最终答案的重要程度（1-5，5表示最关键）  
   - `extra_info`：附加信息或说明（可为空）  
2. 子任务索引 `index` 严格递增，表示执行顺序。  
3. 子任务必须直接服务于最终任务目标，不得加入无关内容。  
4. 输出结构为：

```json
{
  "finish": false,
  "taskItems": [
    {
      "index": "1",
      "task": "任务内容",
      "goal": "任务目标",
      "desc": "任务详细描述或说明",
      "correlation": "相关性数值（1-5）",
      "extra_info": "附加说明或信息"
    }
  ]
}
```

- `finish`：布尔值，表示子任务制定是否完成。  
  - `true`：所有子任务制定完成，根据子任务执行结果可以推导最终答案，无后续子任务。  
  - `false`：还有后续子任务需要动态生成。  
- `taskItems`：子任务列表，每个子任务独立可执行。  

---

### **示例**

用户需求：“分析给定文本中的关键词并统计频率”。

```json
{
  "finish": false,
  "taskItems": [
    {
      "index": "1",
      "task": "提取文本中的所有关键名词",
      "goal": "生成待分析的核心关键词列表",
      "desc": "从背景知识专家提供的文本中，找出对最终分析最相关的名词和术语，注意名词必须准确提取",
      "correlation": "5",
      "extra_info": "确保文本中的专有名词、术语和重要实体都被提取"
    },
    {
      "index": "2",
      "task": "对提取的关键词进行分类和标签化",
      "goal": "将关键词按照主题或类别进行归类",
      "desc": "根据第一步提取的关键词，给每个关键词打上主题标签，以便后续统计和分析",
      "correlation": "4",
      "extra_info": ""
    },
    {
      "index": "3",
      "task": "计算每个关键词在文本中出现的频率",
      "goal": "生成关键词统计表",
      "desc": "统计每个关键词在文本中的出现次数，生成可用于分析的频率表",
      "correlation": "5",
      "extra_info": "确保结果格式可直接用于后续数据分析"
    }
  ]
}
```
""")


class PlannerAgent(BaseAgent):
    def plan(self, state: AgentState) -> AgentState:
        state.node = 'plan'

        print(f"[PLANNER] 计划步骤数: {len(state.task_list)}")

        # 如果是第一次规划，分解任务
        print("[PLANNER] 首次规划，分解任务")
        if state.messages.__len__() <= 0:
            state.messages = [HumanMessage(
                content=f"以下是其他专家给出的背景知识:\n{state.knowledge}\n--- \n*用户任务*：{state.user_task}")]
        plan_result = self._decompose_task([systemMessage] + state.messages)
        state.task_list = plan_result.get("task_list", [])
        state.finished = plan_result.get("finish", [])
        state.tool_calls = plan_result.get("tool_calls", [])
        state.response = plan_result.get("content", '')
        state.current_step_idx = 0
        print(f"[PLANNER] 任务规划结果：{state.task_list}")
        return state

    def _decompose_task(self, messages: List[Any]) -> Dict:
        """
        调用 deepseek 大模型自动分解任务。
        返回结构包含content、tool_calls等
        """
        response = self.invoke_llm(messages,tools=[])

        print("[DECOMPOSE_TASK] response:", response)
        # 提取content
        if hasattr(response, "content"):
            content = getattr(response, "content", "")
        elif isinstance(response, dict):
            content = response.get("content", "")
        else:
            content = ""
        # 提取task_list
        task_list = []
        finish = False
        match = re.search(r'```json\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                loads = json.loads(match.group(1))
                finish = loads.get('finish', False)
                task_list = loads.get('taskItems', False)
            except Exception as e:
                print(f"[DECOMPOSE_TASK] 解析task_list失败: {e}")
        # 返回结构中带task_list
        return {
            "task_list": task_list,
            "finish": finish,
            "response": response,
            "content": content
        }
