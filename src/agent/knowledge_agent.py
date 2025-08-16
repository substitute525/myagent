from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, FunctionMessage

from .base_agent import BaseAgent
from .agent_state import AgentState
from src.tools import all_tools
from .tool_agent import ToolAgent

system_prompt = """你是多智能体系统中的“前置知识智能体”，负责解析用户任务并输出一份结构化背景知识文档，供后续智能体使用。
当前系统结构包括：
你(前置知识智能体): 构建系统背景知识；
PlannerAgent（任务规划智能体）：根据你的输出，制定任务计划；
ExecutorAgent（执行智能体）：根据计划执行具体子任务；
ReviewerAgent（评审总结智能体）：判断任务是否完成，并反馈或总结。
请基于用户任务，输出结构清晰、内容准确的背景知识，以帮助后续智能体高效协作。

**注意**
- 如果你因为某些原因不知道如何处理，可以调用human_assistance工具

请以 Markdown 格式输出，结构如下：
```markdown
# 背景知识
- 主题：
- 涉及领域：
- 已知条件：
- 潜在难点：
```

"""


class KnowledgeAgent(BaseAgent):

    def __init__(self):
        self.tool_node = ToolAgent(tools=all_tools)

    def acquire_knowledge(self, state: AgentState) -> AgentState:
        state.node = 'knowledge'

        # 1. 先用 LLM 获取 tool_calls
        messages = [SystemMessage(content=system_prompt)]
        messages.append(
            HumanMessage(content=f"请为任务提供前置知识或必要准备，并调用工具获取相关信息：\n任务：{state.user_task}"))
        while True:
            response = self.invoke_llm(messages)
            print(
                f"[KNOWLEDGE] 前置知识获取结果: content='{getattr(response, 'content', '')}' additional_kwargs={getattr(response, 'additional_kwargs', {})}")
            tool_calls = getattr(response, 'tool_calls', None)
            if not tool_calls or len(tool_calls) == 0:
                # 用 AIMessage 追加 assistant 回复
                messages.append(AIMessage(content=getattr(response, 'content', '')))
                break
            # 用 AgentState 执行 tool_calls
            tmp_state = AgentState(user_task=state.user_task, execution_results=[], tool_calls=tool_calls)
            tmp_state = self.tool_node(tmp_state)
            # 将每个 tool_call 的结果补充为 ToolMessage
            tool_messages = []
            for result in tmp_state.execution_results:
                if isinstance(result, dict):
                    tool_messages.append(ToolMessage(
                        content=str(result.get('result', '')),
                        name=result.get('tool_name', ''),
                        tool_call_id=result.get('tool_id', '')
                    ))
                else:
                    # 兼容万一 result 已经是 ToolMessage
                    tool_messages.append(result)
            # 用 AIMessage 追加 assistant 回复
            if len(getattr(response, 'content', '')) > 0:
                messages.append(AIMessage(content=getattr(response, 'content', '')))
            messages = messages + tool_messages
        # 可选：将知识内容写入 state
        state.knowledge = getattr(response, 'content', '')
        return state
