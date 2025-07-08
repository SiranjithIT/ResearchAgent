from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import TypedDict, List, Union, Literal
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from datetime import datetime
from dotenv import load_dotenv
from utils import tools

load_dotenv()
memory = MemorySaver()
search = TavilySearch(max_results=5)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]

def call_model(state: AgentState) -> AgentState:
    """Call the model with tools bound."""
    messages = state["messages"]
    
    processed_messages = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            processed_messages.append(HumanMessage(content=f"Interpret this Tool result in proper format to the user: {msg.content}"))
        else:
            processed_messages.append(msg)
    
    model_with_tools = llm.bind_tools(tools = tools + [search])
    #print(f"Inside the call model: {processed_messages}")
    
    try:
        response = model_with_tools.invoke(processed_messages)
        return {"messages": messages + [response]}
    except Exception as e:
        print(f"Error calling model: {e}")
        fallback_response = AIMessage(content=f"The current date and time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return {"messages": messages + [fallback_response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine whether to continue or end the conversation."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "__end__"

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": END,
    },
)
workflow.add_edge("tools", "agent")

agent = workflow.compile(checkpointer=memory)