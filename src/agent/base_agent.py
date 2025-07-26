from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage
from langchain_core.runnables.utils import Output

from src.agent.agent_state import ModelMode
from src.llm import qwen_llm, remote_llm
from typing import List, Optional


@dataclass
class Response:
    content: Optional[str] = ''
    think: Optional[str] = ''
    tool_calls: List = field(default_factory=list)

@dataclass
class BaseAgent():
    mode: ModelMode = ModelMode.LOCAL_QWEN

    def invoke_llm(self, messages: List[BaseMessage]) -> Output:
        if self.mode == ModelMode.LOCAL_QWEN:
            role_map = {
                HumanMessage: "user",
                AIMessage: "assistant",
                SystemMessage: "system",
                FunctionMessage: "function",
                ToolMessage: "function"
            }
            result = []
            for msg in messages:
                role = role_map.get(type(msg), "unknown")
                result.append({
                    "role": role,
                    "content": msg.content,
                    "name": msg.name
                })
            msgs = qwen_llm.run_nonstream(messages=result)
            res = Response()
            for msg in msgs:
                raw = msg.get("content", "")
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = raw.find(start_tag)
                end_idx = raw.find(end_tag)
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    think = raw[start_idx + len(start_tag):end_idx].strip()
                    content = raw[end_idx + len(end_tag):].lstrip()
                else:
                    think = ""
                    content = raw.strip()
                if len(res.content) > 0 and len(content) > 0:
                    print("[重复content]模型返回了多个content")
                if len(content) > 0:
                    res.content = content

                if len(res.think) > 0 and len(think) > 0:
                    print("[重复think]模型返回了多个think")
                if len(think) > 0:
                    res.think = think
                if msg.get('function_call'):
                    res.tool_calls = res.tool_calls + [{
                        "name": msg["function_call"]["name"],
                        "id": msg["function_call"]["name"],
                        "args": msg["function_call"]["arguments"],
                    }]
            return res
        elif self.mode == ModelMode.REMOTE_DS:
            return remote_llm.invoke(messages)

