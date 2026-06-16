import os
from json import tool
from tkinter.filedialog import Open

from dotenv import load_dotenv
from langchain.chat_models import  init_chat_model
from langchain.agents import create_agent
from openai import OpenAI
from openai.types.admin.organization.projects.service_account_create_response import APIKey

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

model = init_chat_model(
    model="OPENAI_API_KEY",
    model_provider="qwen3.7-plus"
)

@tool
def get_weather(city:str)->str:
    """Get weather for a given city."""
    return f"{city}当前天气:晴天,25℃"

agent = create_agent(
    model,
    tools=[get_weather],

)