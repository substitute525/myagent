from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
from langchain_core.messages import ToolMessage


@dataclass
class TaskItem:
    index: int
    task: str
    goal: str
    desc: str
    correlation: int
    extra_info: str


@dataclass
class AgentState:
    user_task: str
    node: str = ''
    execution_results: List[ToolMessage] = field(default_factory=list)
    review_result: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    tool_calls: List[Any] = field(default_factory=list)
    # 背景知识
    knowledge: Any = ""
    # planner相关
    task_finish: bool = False
    task_list: List[TaskItem] = field(default_factory=list)

    messages: List[Any] = field(default_factory=list)
    response: Any = ""

class ModelMode(Enum):
    REMOTE_DS = "remote_deepseek",
    LOCAL_QWEN = "local_qwen"