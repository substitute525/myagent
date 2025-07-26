from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
from langchain_core.messages import ToolMessage

@dataclass
class AgentState:
    user_task: str
    node: str = ''
    execution_results: List[ToolMessage] = field(default_factory=list)
    review_result: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    tool_calls: List[Any] = field(default_factory=list)
    knowledge: Any = ""
    task_list: list = field(default_factory=list)
    messages: List[Any] = field(default_factory=list)
    response: Any = ""

class ModelMode(Enum):
    REMOTE_DS = "remote_deepseek",
    LOCAL_QWEN = "local_qwen"