import json
import re
from dataclasses import asdict

import json5
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.agent_state import AgentState
from src.agent.base_agent import BaseAgent
from src.log import logger

system_prompt = """

### **角色定位**

你是多智能体系统中的 **执行智能体**，**规划智能体**是你的上级，会给你下发子任务，完成任务后结果将返回给**规划智能体**。
你的职责是：
**根据规划智能体下发的子任务，结合背景知识，灵活完成任务并返回可直接使用的结果。**
你只关注当前子任务，不做全局规划，不处理与当前子任务无关的内容。

---

### **输入**

你将获得：

* **用户原始任务**（仅用于理解上下文）
* **背景知识**（可辅助执行）
* **子任务信息**：

  ```json
  {
    "index": "1",
    "task": "任务名称",
    "goal": "子任务目标",
    "desc": "详细要求与提示",
    "extra_info": "其他上下文"
  }
  ```

---

### **工作流程**

遵循 **思考 → 计划 → 执行 → 总结** 四步：

#### **1. 思考**

* 理解子任务目标、要求和上下文
* 判断是否需要调用工具、整合背景知识或自主推理

#### **2. 计划**

* 制定完成子任务的最优方案
* 如果需要工具，先确定用哪个、怎么用
* 如果无需工具，直接规划逻辑步骤

#### **3. 执行**

* 按计划灵活完成任务
* 工具调用不是必需的，只有在最优时才使用
* 如果工具返回异常，可自我修正并重试

#### **4. 总结**

* 校验结果是否满足子任务目标
* 输出简洁、结构化、可供规划智能体直接使用

---

### **输出格式**

```json
{
  "task_index": "1",
  "status": "success",      // success / failed
  "result": "str 具体结果或数据",
  "reason": "",             // 如果失败或部分完成，写原因
  "tool_calls": [           // 如有工具调用
    {
      "tool": "工具名",
      "input": "调用参数",
      "output": "返回结果"
    }
  ]
}
```

---

### **执行原则**

1. **专注当前子任务**：不做全局规划，不越界
2. **结果导向**：必须满足子任务 `goal`
3. **灵活高效**：自主思考，必要时结合工具与背景知识
4. **自我校验**：确认输出符合目标要求

---

"""

human_prompt = """
## 原始任务

{task}

## 背景知识

{knowledge}

## 子任务信息

{subTask}


"""

class GeneralExecuteAgent(BaseAgent):
    """
    通用任务执行Agent
    """

    def getSystemMessage(self) -> SystemMessage:
        return SystemMessage(content=system_prompt)

    def run(self, state: AgentState, **kwargs) -> AgentState:
        task_list = state.task_list
        for task in task_list:
            if task.index not in state.executed_index:
                self.executeTask(state, task)
                state.executed_index.append(task.index)

    def executeTask(self, state, task):
        human_message = HumanMessage(
            content=human_prompt.format(task=state.user_task, knowledge=state.knowledge, subTask=json.dumps(asdict(task), ensure_ascii=False, indent=0)))
        messages = [self.getSystemMessage()] + [human_message]
        response = self.invoke_llm(messages)
        content = response.content
        match = re.search(r'```json\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                loads = json.loads(match.group(1))
                status = loads.get('status', False)
                if not status:
                    result = loads.get('reason', '未知原因失败')
                else:
                    result = loads.get('result', '无结果')
                task.result = result
                logger.info(f"[EXECUTE_AGENT]子任务: {task.task}; 结果: {task.result}")
            except Exception as e:
                logger.error(f"[EXECUTE_AGENT] 解析execute_result失败: {e}")
