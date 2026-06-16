import os
import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from typing import TypedDict,List,Optional


load_dotenv()
model = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

class AgentState(TypedDict):
    message:List[dict]          #对话历史
    current_input:str           #当前用户输入
    tool_name:Optional[str]     #要调用的工具
    tool_args:Optional[dict]    #工具参数
    tool_result: Optional[str]  #工具返回结果
    error:Optional[str]         #错误信息

#Planner
def planner_node(state:AgentState):
    user_input = state["current_input"]
    prompt= f"""
            你是一个Agent决策器。
            用户输入：{user_input}
            请判断：
            1. 是否需要调用工具
            2. 如果需要，选择 tool_name 和参数
            
            可用工具：
            - weather(city)
            
            只返回以下JSON格式,不需要解释：
            {{
              "tool_name": "...",
              "tool_args": {{"city": "..."}}
            }}
            """
    response = model.invoke(prompt)
    content = response.content

    try:
        parsed = json.loads(content)
        return {
            "tool_name":parsed.get("tool_name"),
            "tool_args":parsed.get("tool_args",{})
        }
    except Exception as e :
        return {"error":f"LLM输出解析失败:{str(e)}"}

