from dataclasses import dataclass, field
from typing import List, Optional, Union, Iterator

from langchain_core.messages import BaseMessage, ToolMessage

from src.agent.agent_state import ModelMode
from src.assistant.assistant import BaseAssistant
from src.assistant.qwen_assistant import QwenAssistant
from src.tools import all_tools


@dataclass
class Response:
    content: Optional[str] = ''
    think: Optional[str] = ''
    tool_calls: dict = field(default_factory=dict)


@dataclass
class BaseAgent():
    mode: ModelMode = ModelMode.LOCAL_QWEN
    llm: BaseAssistant = None
    stream: bool = False

    def __init__(self, mode: ModelMode, llm: BaseAssistant = None, llm_cfg: dict = {}, system_prompt: str = None,
                 stream: bool = False):
        if llm_cfg is None:
            llm_cfg = {}
        if llm:
            self.llm = llm
            return
        self.mode = mode
        self.stream = stream
        if mode == ModelMode.LOCAL_QWEN:
            default_cfg = {
                # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
                'model': 'qwen',
                'base_url': 'http://localhost:8000/v1',
                'api_key': 'EMPTY',

                # （可选） LLM 的超参数：
                'generate_cfg': {
                    'top_p': 0.8,
                    'extra_body': {
                        'chat_template_kwargs': {'enable_thinking': True}
                    },
                }
            }
            if llm_cfg:
                llm_cfg = {**default_cfg, **llm_cfg}
            else:
                llm_cfg = default_cfg
            self.llm = QwenAssistant(function_list=all_tools, llm=llm_cfg, name='qwen', system=system_prompt)

    def invoke_llm(self, messages: List[BaseMessage], tools: [str] = None) -> Union[Response, Iterator[Response]]:
        if self.stream:
            response = self.llm.run(messages, stream=self.stream)
            r = Response()
            # TODO: yield
            # yield r
        else:
            r = Response()
            raw = ''
            fn_call = {}
            fn_resp = {}
            for msg_batch in self.llm.run(messages, stream=self.stream, tool_names=tools):
                for response in msg_batch:
                    if isinstance(response, ToolMessage):
                        fn_resp[response.tool_call_id] = response
                    elif hasattr(response, "tool_calls") and response.tool_calls:
                        for tool_call in response.tool_calls:
                            if hasattr(tool_call, "id") and tool_call.id:
                                fn_call[tool_call.id] = response
                    raw = response.content
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
            r.think = think
            r.content = content
            for call_id in set(fn_call) | set(fn_resp):  # 并集，保证两个字典都能覆盖
                r.tool_calls[call_id] = [
                    fn_call.get(call_id),
                    fn_resp.get(call_id),
                ]
            return r
