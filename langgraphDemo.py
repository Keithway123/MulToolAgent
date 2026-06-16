#for langgraph practice
from typing import TypedDict
from langgraph.graph import StateGraph

class MyState(TypedDict):
    a:int
    b:int

def node1(state:MyState):
    return {"a":"1"}
def node2(state:MyState):
    return {"b":"2"}
def node3(state:MyState):
    return {"a":"999"}

builder = StateGraph(MyState)
builder.add_node("node1",node1)
builder.add_node("node2",node2)
builder.add_node("node3",node3)

builder.set_entry_point("node1")

builder.add_edge("node1","node2")
builder.add_edge("node2","node3")

graph1 = builder.compile()

result = graph1.invoke({
        "a":0,
        "b":0
    }
)
print(result)
