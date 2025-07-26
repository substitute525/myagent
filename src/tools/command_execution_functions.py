from typing import Dict, List, Optional
import subprocess
import threading
import uuid
from langchain_core.tools import tool

# 全局 session 管理
_sessions: Dict[str, Dict] = {}
_sessions_lock = threading.Lock()


@tool
def execute_command(command: str, path: str, sessionid: str = None, timeout: int = 10, new_session: bool = True) -> dict:
    """
    使用shell脚本执行命令。new_session默认为True，会自动新建session，若需使用之前的会话则指定sessionid。
    :param sessionid: 指定会话 ID
    :param path: 命令执行的上下文路径
    :param command: 要执行的命令
    :param timeout: 超时时间（秒）
    :param new_session: 是否新建session, 默认为True，若为False，则必须给出一个sessionid
    :return: {"sessionid":当前会话的id, "output”:控制台的输出结果}
    """
    import os
    result = {}
    try:
        shell = True
        if new_session:
            # 新建session
            sessionid = _create_session(shell="bash")
        if not sessionid:
            return {"error": "sessionid不能为空，除非new_session为True"}
        with _sessions_lock:
            if sessionid not in _sessions:
                return {"sessionid": sessionid, "output": [f"[Error] sessionid {sessionid} 不存在"]}
            shell_type = _sessions[sessionid].get('shell', 'cmd')

        # 根据shell类型选择shell参数
        if shell_type == 'cmd':
            shell_executable = None  # 默认shell
        elif shell_type == 'powershell':
            shell_executable = 'powershell'
        elif shell_type == 'bash':
            shell_executable = r'D:\工具\编程\git\bin\bash.exe'
            command = ['/usr/bin/shell', '-c'] + [command]
            shell = False
        else:
            shell_executable = None  # fallback

        # 执行命令
        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            executable=shell_executable,
            encoding='utf-8'
            # creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        try:
            outs, _ = proc.communicate(timeout=timeout, input='ls')
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, _ = proc.communicate()
            outs += '\n[Timeout]'
            timed_out = True
        output_lines = outs.splitlines()
        with _sessions_lock:
            _sessions[sessionid]['output'].extend(output_lines)
            _sessions[sessionid]['last_command'] = command
            _sessions[sessionid]['history'].append(command)
        result = {"sessionid": sessionid, "output": output_lines, "timeout": timed_out}
    except Exception as e:
        with _sessions_lock:
            _sessions[sessionid]['output'].append(f'[Error] {e}')
            _sessions[sessionid]['last_command'] = command
            _sessions[sessionid]['history'].append(command)
        result = {"error": str(e)}
    return result


@tool
def list_sessions() -> List[Dict]:
    """
    查询存活的 session。
    :return: [{sessionid, last_command, shell}]
    """
    with _sessions_lock:
        return [
            {'sessionid': sid, 'last_command': sess['last_command'], 'shell': sess.get('shell', 'cmd')}
            for sid, sess in _sessions.items()
        ]


@tool
def read_output(sessionid: str, lines: int = 10) -> List[str]:
    """
    读取指定 session 的输出。
    :param sessionid: 会话 ID
    :param lines: 需要读取的行数
    :return: 输出内容（行）
    """
    with _sessions_lock:
        if sessionid not in _sessions:
            return [f'[Error] sessionid {sessionid} 不存在']
        output = _sessions[sessionid]['output']
        return output[-lines:] if lines > 0 else output


@tool
def create_session(shell: str = 'cmd') -> str:
    """
    创建一个新的 sessionid
    :param shell: 指定shell类型，支持cmd、powershell、bash，默认cmd
    """
    return _create_session(shell)

# @tool
def _create_session(shell: str = 'cmd') -> str:
    sessionid = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[sessionid] = {
            'history': [],
            'output': [],
            'last_command': None,
            'shell': shell
        }
    return sessionid

if __name__ == '__main__':
    command = execute_command(command='ls', path='S:\\\\work')
    print(command)