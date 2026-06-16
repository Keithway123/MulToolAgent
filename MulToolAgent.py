import os
import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from typing import TypedDict,List,Optional

from langchain_core.prompts import prompt
from langgraph.graph import StateGraph


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

#Planner_Node
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
        data = json.loads(content)
        return {
            "tool_name":data.get("tool_name"),
            "tool_args":data.get("tool_args",{})
        }
    except Exception as e :
        return {"error":f"LLM输出解析失败:{str(e)}"}

#Tool
def weather_tool(city: str):
    return f"{city}天气晴，25℃"

#Tool_Node
def tool_node(state:AgentState):
    if state.get("error"):
        return  state

    tool_name = state.get("tool_name")

    if not tool_name:
        return {}

    try:
        if tool_name == "weather":
            result = weather_tool(**state["tool_args"])
            return {"tool_result":result}

        return{"error":"未知工具"}

    except Exception as e:
        return {"error": str(e)}

#response_node
def response_node(state:AgentState):
    if state.get("error"):
        return{
            "message":[
                {"role":"assistant", "content":state["error"]}
            ]
        }

    if state.get("tool_result"):
        return{
            "message":[
                {"role":"assistant", "content":state["tool_result"]}
            ]
        }

    return{
        "message":[
            {"role":"assistant", "content":"不需要调用工具"}
        ]
    }

#final_node
def final_node(state:AgentState):
    prompt = f"""
        你是一个助手。
        用户输入：{state["current_input"]}
        工具返回：{state.get("tool_result","{}")}
        请给出你的回复。
        """
    response = model.invoke(prompt)
    return {
        "message":[
            {"role":"assistant", "content":response.content}
        ]
    }

#Grpah
builder = StateGraph(AgentState)
builder.add_node("planner_node",planner_node)
builder.add_node("tool_node",tool_node)
builder.add_node("response_node",response_node)
builder.add_node("final_node",final_node)

builder.set_entry_point("planner_node")

builder.add_edge("planner_node","tool_node")
builder.add_edge("tool_node","final_node")

graph = builder.compile()
result = graph.invoke({
    "message":[],
    "current_input":"你是谁？"
})

print(result)