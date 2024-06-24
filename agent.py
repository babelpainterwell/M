from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai) # functional node that calls the model, call the state by default?
        graph.add_node("action", self.take_action) # functional node that searches the web,
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
    
    def exists_action(self, state: AgentState):
        print("****** Checking if action exists ******")
        result = state["messages"][-1] # get the AI response and check if there is tool calls
        return len(result.tool_calls) > 0 # so if in AI response, there are no tool calls, then return False
    
    def call_openai(self, state: AgentState): # takes current state as input and returns the updated messages
        print("****** Calling OpenAI ******")
        messages = state["messages"] # get the current messages 
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages # append the system message to the messages
        updated_messages = self.model.invoke(messages)
        return {'messages': [updated_messages]} # doesn't add the system message 

    # human input should be added to the take_action function
    def take_action(self, state:AgentState):
        print("****** Taking action ******")
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for call in tool_calls:
            print(f"Calling: {call}")
            # use the tool, also by using invoke
            result = self.tools[call['name']].invoke(call['args'])
            # results.append(ToolMessage(tool_call_id=call['id'], name=call['name'],content=str(result)))
            response = ToolMessage(tool_call_id=call['id'], name=call['name'], content=str(result))
            print("Back to the agent")
        return {'messages': response} # why here is not a single message


def main():
    _ = load_dotenv()

    model = ChatOpenAI(model="gpt-4o")
    tool = TavilySearchResults(max_results=5)
    print(type(tool))
    print(tool.name)

    system_prompt = """ You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    agent = Agent(model, [tool], system=system_prompt)

    # draw the graph 
    # from IPython.display import Image
    # Image(agent.graph.get_graph().draw_png())

    messages = [HumanMessage(content="Males in which country are the tallest")]
    result = agent.graph.invoke({"messages": messages})

    # print(result)
    # print("*" * 20)
    print(result['messages'])
    print(result["messages"][1].tool_calls)
    # for m in result['messages']:
    #     print(m)
    





if __name__ == "__main__":
    main()