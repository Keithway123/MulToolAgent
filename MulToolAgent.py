import os
from json import tool

from dotenv import load_dotenv
from langchain.chat_models import  init_chat_model
from langchain.agents import create_agent
from langchain.tools import  tool
from langchain.messages import AIMessage
from openai.resources.chat.completions import messages

load_dotenv()

model = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("DASHSCOPE_BASE_URL")
)

@tool
def get_weather(city:str)->str:
    """Get weather for a given city."""
    return f"{city}当前天气:晴天,25℃"

system_prompt = """
You are a mul Tool assistant.
Your Name is Qwen man .
"""

agent = create_agent(
    model,
    tools=[get_weather],
    system_prompt=system_prompt
)

response = agent.invoke(
    {"message":[("user", "今天东莞市天气如何？")]}
)

print(response["messages"][-1].content)
