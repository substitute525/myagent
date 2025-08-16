import inspect
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_openai import ChatOpenAI

LLM_REGISTRY = {}


class OpenAi(ChatOpenAI):
    def __init__(self, cfg: dict):
        allowed_params = ['timeout', 'max_retries','api_key','base_url','organization','max_tokens','temperature','model']
        # 筛选
        filtered_kwargs = {k: v for k, v in cfg.items() if k in allowed_params}
        super().__init__(**filtered_kwargs)

LLM_REGISTRY['oai'] = OpenAi

def convert_messages_to_openai(messages: List[BaseMessage]):
    hf_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            hf_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            hf_messages.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            hf_messages.append({"role": "system", "content": m.content})
        elif isinstance(m, FunctionMessage):
            hf_messages.append({"role": "tool", "content": m.content})
        else:
            raise ValueError(f"Unsupported message type: {m}")
    return hf_messages