import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from command_execution_functions import create_session, execute_command, read_output, list_sessions
from core.tools_registry import human_assistance

def test_command_execution_ls():
    # 1. 创建 session，默认shell
    sessionid = create_session.invoke({})
    assert sessionid is not None
    print(f"创建的sessionid: {sessionid}")

    # 2. 执行ls命令（Windows下用dir，Linux/Mac下用ls）
    test_path = os.getcwd()
    command = 'ls' if os.name != 'nt' else 'dir'
    execute_command.invoke({
        'sessionid': sessionid,
        'path': test_path,
        'command': command,
        'timeout': 5
    })

    # 3. 等待命令执行完成
    time.sleep(2)

    # 4. 读取输出
    output = read_output.invoke({'sessionid': sessionid, 'lines': 20})
    print("命令输出:")
    for line in output:
        print(line)
    assert any('py' in line or 'mermaid' in line for line in output)

    # 5. 查询session
    sessions = list_sessions.invoke({})
    print("当前session列表:", sessions)
    assert any(s['sessionid'] == sessionid for s in sessions)

    # 6. 在同一个session中再执行一条命令
    command2 = 'echo test_second_command' if os.name != 'nt' else 'echo test_second_command'
    execute_command.invoke({
        'sessionid': sessionid,
        'path': test_path,
        'command': command2,
        'timeout': 5
    })
    time.sleep(1)
    output2 = read_output.invoke({'sessionid': sessionid, 'lines': 10})
    print("第二条命令输出:")
    for line in output2:
        print(line)
    assert any('test_second_command' in line for line in output2)

    # 7. 再次确认session仍然存在且命令历史正确
    sessions2 = list_sessions.invoke({})
    found = False
    for s in sessions2:
        if s['sessionid'] == sessionid:
            found = True
            print(f"session {sessionid} 最后命令: {s['last_command']}")
            assert s['last_command'] == command2
    assert found

    # 8. 创建 powershell session（仅Windows测试）
    if os.name == 'nt':
        ps_sessionid = create_session.invoke({'shell': 'powershell'})
        assert ps_sessionid is not None
        print(f"创建的powershell sessionid: {ps_sessionid}")
        execute_command.invoke({
            'sessionid': ps_sessionid,
            'path': test_path,
            'command': command,
            'timeout': 5
        })
        time.sleep(2)
        output_ps = read_output.invoke({'sessionid': ps_sessionid, 'lines': 20})
        print("powershell命令输出:")
        for line in output_ps:
            print(line)
        assert any('py' in line or 'mermaid' in line for line in output_ps)
        sessions = list_sessions.invoke({})
        assert any(s['sessionid'] == ps_sessionid and s['shell'] == 'powershell' for s in sessions)

def test_execute_command_new_session():
    command = 'echo hello_test' if os.name == 'nt' else 'echo hello_test'
    result = execute_command.invoke({
        'command': command,
        'path': os.getcwd(),
        'timeout': 5,
        'new_session': True
    })
    print('new_session result:', result)
    assert 'output' in result and any('hello_test' in line for line in result['output'])

def test_human_assistance():
    try:
        result = human_assistance.invoke({"query": "请帮我人工处理一个特殊问题"})
        print("human_assistance result:", result)
        assert isinstance(result, str) or (isinstance(result, dict) and "data" in result)
    except Exception as e:
        print("human_assistance 调用异常：", e)
        assert False

if __name__ == "__main__":
    test_command_execution_ls()
    test_execute_command_new_session()
    test_human_assistance() 