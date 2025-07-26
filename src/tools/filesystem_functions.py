import os
from typing import List, Dict

from langchain_core.tools import tool


@tool
def list_dir(path: str) -> List[str]:
    """
    查询指定目录下的文件和目录。
    :param path: 目录路径
    :return: 文件和目录名列表
    """
    try:
        return os.listdir(path)
    except Exception as e:
        return [f"[Error] {e}"]

@tool
def read_file_lines(path: str, start: int, end: int) -> Dict:
    """
    读取指定文件的部分内容。
    :param path: 文件路径
    :param start: 开始行（1为首行）
    :param end: 结束行（包含）
    :return: {'lines': [内容], 'total_lines': 总行数}
    """
    lines = []
    total = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            total = len(all_lines)
            # 行号从1开始
            lines = all_lines[start-1:end]
        return {'lines': [l.rstrip('\n') for l in lines], 'total_lines': total}
    except Exception as e:
        return {'lines': [f"[Error] {e}"], 'total_lines': total}

@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """
    写入文件。
    :param path: 文件路径
    :param content: 写入内容
    :param append: 是否追加，默认覆盖
    :return: 写入结果
    """
    try:
        mode = 'a' if append else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        return 'success'
    except Exception as e:
        return f"[Error] {e}"

@tool
def delete_file(path: str) -> str:
    """
    删除文件。
    :param path: 文件路径
    :return: 删除结果
    """
    try:
        os.remove(path)
        return 'success'
    except Exception as e:
        return f"[Error] {e}"

@tool
def replace_in_file(path: str, old: str, new: str) -> str:
    """
    替换文件中的指定文本（可多行）。
    :param path: 文件路径
    :param old: 需要替换的内容
    :param new: 替换为的内容
    :return: 替换结果
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        content_new = content.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content_new)
        return 'success'
    except Exception as e:
        return f"[Error] {e}"

@tool
def tree_dir(path: str, exclude_dirs: list = None) -> List[str]:
    """
    递归列出目录树，排除指定目录和隐藏文件夹。
    :param path: 根目录
    :param exclude_dirs: 不需要展开的目录名称列表，默认不展开隐藏文件和打包文件，如无特殊要求，禁止传入该参数
    :return: 目录树字符串列表
    """
    if exclude_dirs is None:
        exclude_dirs = ["target", "node_modules", "dist", "build", "__pycache__", ".git", ".idea"]
    tree = []
    def _tree(current_path, prefix=""):
        try:
            entries = os.listdir(current_path)
        except Exception as e:
            tree.append(prefix + f"[Error] {e}")
            return
        entries = sorted(entries)
        for idx, entry in enumerate(entries):
            full_path = os.path.join(current_path, entry)
            is_last = idx == len(entries) - 1
            connector = "└── " if is_last else "├── "
            tree.append(prefix + connector + entry)
            # 排除隐藏文件夹和指定目录
            if os.path.isdir(full_path):
                if entry.startswith(".") or entry in exclude_dirs:
                    continue
                _tree(full_path, prefix + ("    " if is_last else "│   "))
    _tree(path)
    return tree 