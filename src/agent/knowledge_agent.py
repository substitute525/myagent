import re

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_state import AgentState
from .base_agent import BaseAgent

system_prompt = """# 职责说明

你是多智能体系统中的“前置知识智能体”，负责解析用户任务并输出一份结构化背景知识文档，供后续智能体使用。
当前系统结构包括：
你(前置知识智能体): 构建系统背景知识；
PlannerAgent（任务规划智能体）：根据你的输出，制定任务计划；
ExecutorAgent（执行智能体）：根据计划执行具体子任务；
ReviewerAgent（评审总结智能体）：判断任务是否完成，并反馈或总结。
请基于用户任务，输出结构清晰、内容准确的背景知识，以帮助后续智能体高效协作。

## 注意

- 如果你因为某些原因不知道如何处理，可以调用human_assistance工具

{tools}

# 最终结果
若*需要调用工具*，则先调用工具。
若***不需要调用工具***或***已经调用完成***，则将上下文信息整理为以下格式：
``` markdown
# 背景知识
- 涉及领域：{{根据已知信息分类得出的领域}}
- 已知事实：{{根据所有上下文整理出的与用户问题有关的事实}}
- 潜在难点：{{根据用户问题推理/分析得出的难点或需注意的点}}

```
仅以上 Markdown 作为最终的背景知识
"""


class KnowledgeAgent(BaseAgent):

    def acquire_knowledge(self, state: AgentState) -> AgentState:
        state.node = 'knowledge'

        # 1. 先用 LLM 获取 tool_calls
        messages = [SystemMessage(content=system_prompt)]
        messages.append(
            HumanMessage(content=f"请为任务提供前置知识或必要准备，并调用工具获取相关信息：\n任务：{state.user_task}"))
        response = self.invoke_llm(messages,
                                   tools=['execute_command', 'list_sessions', 'read_output', 'human_assistance',
                                          'tree_dir', 'query_url', 'search_web'])
        print(
            f"[KNOWLEDGE] 前置知识获取结果: think='{response.think}'\n content='{response.content}'\n tool_calls='{response.tool_calls}' ")

        # 将知识内容写入 state
        pattern = r"markdown\n(.*?)"
        matches = re.findall(pattern, response.content, re.DOTALL)

        for m in matches:
            state.knowledge = m
        if not state.knowledge:
            state.knowledge = response.content
        state.tool_calls.extend(response.tool_calls.values())
        return state
