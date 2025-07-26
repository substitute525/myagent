import json
from .agent_state import AgentState
from langchain_core.messages import ToolMessage

from .base_agent import BaseAgent


class ToolAgent(BaseAgent):
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def run_task(self, state: AgentState) -> AgentState:
        state.node = 'tools'
        for task in state.task_list:
            messages = []
            self.invoke_llm();


    def __call__(self, state: AgentState) -> AgentState:
        state.node = 'tools'

        tool_calls = getattr(state, 'tool_calls', None)
        if not tool_calls:
            print("[TOOL_AGENT] 没有工具调用")
            return state
        print(f"[TOOL_AGENT] 需要调用: {len(tool_calls)} 个工具")
        outputs = []
        tool_messages = []
        for tool_call in tool_calls:
            try:
                if isinstance(tool_call, dict):
                    if "function" in tool_call:
                        tool_name = tool_call["function"].get("name")
                        tool_args_str = tool_call["function"].get("arguments", "{}")
                        tool_id = tool_call.get("id")
                    else:
                        tool_name = tool_call.get("name")
                        tool_args_str = tool_call.get("args", "{}")
                        tool_id = tool_call.get("id")
                else:
                    tool_name = getattr(tool_call, "name", None)
                    tool_args_str = getattr(tool_call, "args", "{}")
                    tool_id = getattr(tool_call, "id", None)
                print(f"[TOOL_AGENT] 调用工具: {tool_name}, 参数: {tool_args_str}")
                try:
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except Exception as e:
                    print(f"[TOOL_AGENT] JSON解析错误: {e}, 使用空字典")
                    tool_args = {}
                if not tool_name:
                    print(f"[TOOL_AGENT] 工具名称为空，跳过")
                    continue
                if tool_name in self.tools_by_name:
                    tool_result = self.tools_by_name[tool_name].invoke(tool_args)
                    print(f"[TOOL_AGENT] 工具 {tool_name} 执行结果: {str(tool_result)[:200]}...")
                    outputs.append(ToolMessage(
                        content=str(tool_result),
                        name=tool_name,
                        tool_call_id=tool_id
                    ))
                else:
                    print(f"[TOOL_AGENT] 工具 {tool_name} 不存在")
                    outputs.append(ToolMessage(
                        content=f"工具 {tool_name} 不存在",
                        name=tool_name or "unknown",
                        tool_call_id=tool_id or ""
                    ))
            except Exception as e:
                print(f"[TOOL_AGENT] 工具调用错误: {e}")
                outputs.append(ToolMessage(
                    content=str(e),
                    name=tool_name if 'tool_name' in locals() else "unknown",
                    tool_call_id=tool_id if 'tool_id' in locals() else ""
                ))
        state.execution_results.extend(outputs)
        if hasattr(state, 'messages'):
            state.messages.extend(outputs)
        return state 