from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, FunctionMessage

from .base_agent import BaseAgent
from .agent_state import AgentState
from src.tools import all_tools
from .tool_agent import ToolAgent

system_prompt = """你是一个用于软件开发支持的前置知识分析助手。
用户希望你分析当前任务中可能需要的基础前置信息，例如文件结构、配置、关键代码文件、依赖说明等。并通过调用工具获取必要的信息。
前置信息通常仅作为或许的基础知识背景或处理依据，因此需要提取的是必要的信息，而不需要提取详细、全面的信息。
若用户有明确要求调用工具完成什么命令的，则你可以直接完成。
完成工具调用后，你需要整理所有信息，将其汇总为知识文档。

**注意**
- 如果你因为某些原因不知道如何处理，可以调用human_assistance工具
- 大多数情况你都需要调用list_sessions工具列举出当前存活的session，以供后续使用
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
