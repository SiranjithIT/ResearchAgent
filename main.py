from Agent import agent
from langchain_core.messages import HumanMessage
import json

def create_file(name:str, content:dict)->None:
    """Create a file with the given name and content."""
    with open(name, 'w') as f:
        json.dump(content, f, indent=4)

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
        content = result['messages'][-1].content
        if not isinstance(content, str):
            content = str(content)
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        elif not content.startswith('{') and not content.endswith('}') or not content.startswith('[') and not content.endswith(']'):
            print("Content is not in JSON format.")
            continue

        if not content:
            print("No content received.")
            continue
        content = content.strip()
        choice = input("Do you want to save the result? (yes/no): ").strip().lower()
        if choice == 'yes':
            file_name = input("Enter the file name (default: papers.json): ").strip()
            if not file_name:
                file_name = 'papers.json'
            create_file(file_name+".json", json.loads(content))
            print(f"Result saved to {file_name}")

if __name__ == "__main__":
    main()