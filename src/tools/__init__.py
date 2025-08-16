import json

import langchain_core.tools
from dotenv import load_dotenv
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool, TOOL_REGISTRY

load_dotenv()
from src.tools.command_execution_functions import (
    execute_command,
    list_sessions,
    read_output,
    create_session
)
from src.tools.filesystem_functions import (
    list_dir,
    read_file_lines,
    write_file,
    delete_file,
    replace_in_file,
    tree_dir
)
from src.tools.web_query_functions import (
    query_url,
    search_web
)
from src.tools.interrupt import human_assistance

all_tools = [
    human_assistance,
    execute_command,
    list_sessions,
    read_output,
    create_session,
    list_dir,
    read_file_lines,
    write_file,
    delete_file,
    replace_in_file,
    tree_dir,
    query_url,
    search_web
]


def register_qwen_tool_from_langchain(lc_tool, name=None, description=None):
    ToolCls, tool_name = get_qwen_cls(lc_tool, description, name)

    return register_tool(tool_name)(ToolCls)


def get_qwen_cls(lc_tool: langchain_core.tools.BaseTool, description=None, name=None):
    tool_name = name or lc_tool.name
    tool_description = description or getattr(lc_tool, "description", "")
    parameters = []
    # 如果有 args_schema，用其 schema() 提取字段
    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema:
        schema_info = lc_tool.args_schema.schema()
        required_fields = schema_info.get("required", [])
        for param_name, param_info in schema_info.get("properties", {}).items():
            parameters.append({
                "name": param_name,
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
                "required": param_name in required_fields,
            })

    def tool_call(self, params: str, **kwargs) -> str:
        parsed = json.loads(params) if params else {}
        result = lc_tool.invoke(parsed)
        print(f"[工具调用]工具名称：{lc_tool.name}, 参数：{params}, 结果：{result}")
        return json.dumps({"result": result})

    ToolCls = type(
        tool_name + "Tool",
        (BaseTool,),
        {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters,
            "call": tool_call,
        }
    )
    return ToolCls, tool_name


def qwen_adaptation_init():
    # function适配
    for tool in all_tools:
        if tool.name in TOOL_REGISTRY:
            continue
        register_qwen_tool_from_langchain(tool)
