import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .agent_state import AgentState
from .base_agent import BaseAgent

systemMessage = SystemMessage(content="""
你是一个代码任务的分解与规划专家，你需要拆分任务为子任务，确保最终完成任务目标。
当前你可能已获得的信息：
1. 任务前置知识，例如项目结构、文件内容等。（其他专家给出）；
2. 工具执行结果（如文件读取结果等）；
若以上信息中包含或可以推导出结果，则不进行任务规则，直接返回空数组。
若以上信息不包含结果则你需要输出一轮完整的任务规划，包括：
- 本轮总共要做哪些子任务（该部分必须使用json数组格式）；
- 哪些子任务需要使用哪些工具；

注意事项：
- 若需要创建任务，则确保每一个子任务的必要性，禁止过度拆分任务以及无意义的任务
- 你仅能完成工具调用以及输出任务规划相关的，禁止回答用户任何问题
- 请保持历史任务编号不变，若你需要调整任务，你可以修改还未执行的步骤或继续追加步骤；
- 若你未调用工具，则将视为任务已规划完成。
- 如果你因为某些原因不知道如何处理，可以调用human_assistance工具

输出格式如下：
任务规划说明：
`用自然语言自然语言解释，可包含对前置知识的解释和当前任务的分解`
任务步骤：
```json
[....]
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
        state.tool_calls = plan_result.get("tool_calls", [])
        state.response = plan_result.get("content", '')
        state.current_step_idx = 0
        return state

    def _decompose_task(self, messages: List[Any]) -> Dict:
        """
        调用 deepseek 大模型自动分解任务。
        返回结构包含content、tool_calls等
        """
        response = self.invoke_llm(messages)

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
        match = re.search(r'```json\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                task_list = json.loads(match.group(1))
            except Exception as e:
                print(f"[DECOMPOSE_TASK] 解析task_list失败: {e}")
        # 提取tool_calls
        if hasattr(response, "tool_calls"):
            tool_calls = getattr(response, "tool_calls", None)
        elif isinstance(response, dict):
            tool_calls = response.get("tool_calls")
        else:
            tool_calls = None
        print("[DECOMPOSE_TASK] :", {
            "content": content,
            "tool_calls": tool_calls,
            "task_list": task_list,
            "response": response
        })
        # 返回结构中带task_list
        return {
            "tool_calls": tool_calls,
            "task_list": task_list,
            "response": response,
            "content": content
        }
