from Agent import agent
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

def main():
    config = {"configurable": {"thread_id": "1"}}
    while True:
        input_text = input("User: ")
        if 'exit' in input_text:
            break
        
        result = agent.invoke(
                {"messages": [HumanMessage(content=input_text)]}, 
                config=config
            )
        print(f"AI: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()