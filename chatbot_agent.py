
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import json




"""
For tool calling:

Next we need to create a function to actually run the tools if they are called. 
We'll do this by adding the tools to a new node.

Below, implement a BasicToolNode that checks the most recent message in the state 
and calls tools if the message contains tool_calls. It relies on the LLM's tool_calling` support, 
which is available in Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.
"""

class UserExitError(Exception):
    def __init__(self, message):
        self.message = message


class BotState(TypedDict):
    messages: Annotated[list, add_messages] # the new image will be appended to the list of messages to update the state


class Chatbot:
    def __init__(self, model, system=""):
        self.system = system
        self.model = model 
        self.iter = 1
        graph = StateGraph(BotState) # BotState here is a schema for the state of the agent
        graph.add_node("llm", self.call_openai) # 'llm' here is just a node name 
        graph.add_node("input", self.ask_for_input)
        graph.add_edge("input", "llm")
        graph.set_entry_point("input")
        # graph.set_finish_point("llm") # only execute the loop once
        graph.add_conditional_edges(
            "llm",
            self.continue_loop, # even though in the loop, there is no place for the user to input
            {True: "input", False: END}
        )
        self.graph = graph.compile()
    
    def call_openai(self, state: BotState):
        self.iter += 1
        messages = state['messages']
        if self.system:
            messages = [self.system] + messages
        print(messages)
        response = self.model.invoke(messages) # single message?
        print("Bot:", response.content)
        return {"messages": [response]}
    
    def ask_for_input(self, state: BotState):
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            # customizr a user exit error
            raise UserExitError("User exited the chat")
        return {"messages": [HumanMessage(content=user_input)]}

    def continue_loop(self, state: BotState):
        if self.iter > 5:
            print("I'm tired, goodbye!")
            return False
        return True

class ToolNode:
    """
    A node that runs the tool in the last AI message
    An integration of checking if there is a tool call but also running the tool
    """
    def __init__(self, tools:list):
        self.tools_by_name = {t.name: t for t in tools} # create a dictionary of tools

    def __call__(self, inputs:dict):
        if messages := inputs.get("messages", []): # return an empty as the default value 
            message = messages[-1]
        else:
            raise ValueError("No messages in the input")
        outputs = []
        # tool_call has 'args'
        # json.dump: convert the tool result to a string
        for tool_call in message.tool_calls:
            # find the rool in our tool dictionary according to its name given by the ai message
            tool_result = self.tools_by_name[tool_call['name']].invoke(tool_call['args'])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": [outputs]}

def main():
    model = ChatOpenAI(model="gpt-4o")
    tool = TavilySearchResults(max_results=5)
    tools = [tool]
    # model_with_tools = model.bind_tools(tools)
    chatbot = Chatbot(model=model, system=SystemMessage(content="You are a thoughtful robot"))

    try: 
        result = chatbot.graph.invoke({"messages": []}) # will return 
    except Exception as e:
        if isinstance(e, UserExitError):
            print(e.message)
    # print(result)

    
    # result = chatbot.graph.invoke({"messages": [user_input]})
    # print("Bot:", result['messages'][-1].content)
    # print("*************************************************")

    # print(result['messages'])


    # message = "Hello, how are you?"
    # result = chatbot.graph.invoke({"messages": [message]})

    # print(result['messages'][-1].content)

    # the state doesn't continue without the conditional node
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == "exit":
    #         print("Goodbye!")
    #         break
    #     result = chatbot.graph.invoke({"messages": [user_input]}) 
    #     print("Bot:", result['messages'][-1].content)
    

if __name__ == "__main__":
    main()


