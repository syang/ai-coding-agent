import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.tools import tool
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from IPython.display import Image, display

# Load environment variables
load_dotenv()

# Define the state
class State(dict):
    """The state of the system."""
    def __init__(self):
        self.project_requirements = ""
        self.generated_code = ""
        self.execution_result = ""
        self.debug_history = []
        self.github_repo = ""

# Define tool functions
@tool
def analyze_requirements(input: str) -> str:
    """Analyze and structure project requirements."""
    return f"Analyzed requirements: {input}"

@tool
def generate_code(input: str, current_code: str) -> str:
    """Generate code based on requirements."""
    return f"Generated code for: {input}"

@tool
def debug_code(code: str, error: str) -> dict:
    """Debug and fix code errors."""
    return {"output": f"Debugged {code}", "fixed_code": code}

@tool
def push_to_github(code: str, repo: str) -> str:
    """Push code to GitHub repository."""
    return f"Pushed code to {repo}"

# Node functions
def requirement_analysis(state: State) -> Dict[str, Any]:
    result = project_manager_executor.invoke({
        "input": state.project_requirements
    })
    state.project_requirements = result['output']
    return {"next": "code_generation"}

def code_generation(state: State) -> Dict[str, Any]:
    result = code_generator_executor.invoke({
        "input": state.project_requirements,
        "current_code": state.generated_code
    })
    state.generated_code = result['output']
    return {"next": "code_execution"}

def code_execution(state: State) -> Dict[str, Any]:
    # Simulate code execution
    state.execution_result = "Simulated execution result"
    if "error" in state.execution_result.lower():
        return {"next": "debugging"}
    else:
        return {"next": "github_integration"}

def debugging(state: State) -> Dict[str, Any]:
    result = debugger_executor.invoke({
        "code": state.generated_code,
        "error": state.execution_result
    })
    state.debug_history.append(result['output'])
    state.generated_code = result['fixed_code']
    return {"next": "code_execution"}

def github_integration(state: State) -> Dict[str, Any]:
    result = github_executor.invoke({
        "code": state.generated_code,
        "repo": state.github_repo
    })
    state.github_repo = result['output']
    return {"next": "requirement_update"}

def requirement_update(state: State) -> Dict[str, Any]:
    result = project_manager_executor.invoke({
        "current_requirements": state.project_requirements,
        "new_input": "User's new input here"  # This would come from some user input mechanism
    })
    state.project_requirements = result['output']
    return {"next": "code_generation"}

# Define agents
llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

project_manager_agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm,
    [analyze_requirements],
    system_message=SystemMessage(content="You are a project manager overseeing the development process.")
)
project_manager_executor = AgentExecutor(agent=project_manager_agent, tools=[analyze_requirements])

code_generator_agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm,
    [generate_code],
    system_message=SystemMessage(content="You are a code generator specializing in FastAPI, PostgreSQL, and SQLAlchemy.")
)
code_generator_executor = AgentExecutor(agent=code_generator_agent, tools=[generate_code])

debugger_agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm,
    [debug_code],
    system_message=SystemMessage(content="You are a debugging expert capable of analyzing and fixing code errors.")
)
debugger_executor = AgentExecutor(agent=debugger_agent, tools=[debug_code])

github_agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm,
    [push_to_github],
    system_message=SystemMessage(content="You are responsible for managing GitHub interactions and code pushing.")
)
github_executor = AgentExecutor(agent=github_agent, tools=[push_to_github])

# Create the graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("requirement_analysis", requirement_analysis)
workflow.add_node("code_generation", code_generation)
workflow.add_node("code_execution", code_execution)
workflow.add_node("debugging", debugging)
workflow.add_node("github_integration", github_integration)
workflow.add_node("requirement_update", requirement_update)

# Add edges to the graph
workflow.add_edge("requirement_analysis", "code_generation")
workflow.add_edge("code_generation", "code_execution")
workflow.add_conditional_edges(
    "code_execution",
    lambda x: x["next"],
    {
        "debugging": "debugging",
        "github_integration": "github_integration"
    }
)
workflow.add_edge("debugging", "code_execution")
workflow.add_edge("github_integration", "requirement_update")
workflow.add_edge("requirement_update", "code_generation")

# Set the entry point
workflow.set_entry_point("requirement_analysis")

# Compile the graph
app = workflow.compile()

print(";;;;;;;;;;;;;;;;;;;")
# display(Image(app.get_graph().draw_png()))

print("=============")
app.get_graph().print_ascii()
print(".............")


# Define the function to run the workflow
def run_alex(user_input: str):
    state = State()
    state.project_requirements = user_input
    for step in app:
        # Assuming step is a tuple with at least two elements
        node, output = step
        print(f"Current node: {node}")
        if isinstance(output, dict):
            # Handle output as a dictionary
            print(f"Output: {output}")
            # Update state based on output
            if output.get('type') == 'end':
                return output['output']
        else:
            # Handle output as a string (name of the LangGraph)
            print(f"LangGraph name: {output}")

# Example usage
if __name__ == "__main__":
    user_input = "I need to build a FastAPI service with PostgreSQL as database layer, and I need to use SQLAlchemy as my ORM layer. Please write the project for me. We should have an endpoint of POST /items and GET /items"
    result = run_alex(user_input)
    print("Final result:", result)
