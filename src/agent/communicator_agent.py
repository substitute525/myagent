from src.agent.base_agent import BaseAgent


class CommunicatorAgent(BaseAgent):
    def communicate(self, state):
        print("[COMMUNICATOR] 结果反馈用户：")
        print(f"任务: {state.user_task}")
        print(f"计划: {state.plan_steps}")
        print(f"执行结果: {state.execution_results}")
        print(f"审查结果: {state.review_result}")
        return state 