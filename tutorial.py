
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.pydantic_v1 import BaseModel
import json
import argparse
from prompts import SYSTEM_PROMPT
import logging
import os


"""
For tool calling:

Next we need to create a function to actually run the tools if they are called. 
We'll do this by adding the tools to a new node.

Below, implement a BasicToolNode that checks the most recent message in the state 
and calls tools if the message contains tool_calls. It relies on the LLM's tool_calling` support, 
which is available in Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.
"""

"""
Tips
1. Human feedback can be achieved via update_state to add HumanMessage to the state, acting. 
2. The supervisor model could be repalced by the customizing state functionaity, eg. adding supervisor_mode bool to the state.
3. Add one more attribute 'query_human' to ask human for additional information. 
"""


"""
To-Do
1. Human feedback to agent's action, and the agent will rethink if necessary. 
"""

# Global configuration
config = {"configurable": {"thread_id": "1"}}

# Global system prompt 
SYSTEM_PROMPT = """
                To provide an accurate and concise answer, ask for additional information from the user if necessary. For example, if the user asks, “What is the NBA score?”, you should ask which game they are referring to.
                For interactions that require additional information from humans, use the tool RequestAdditionalInfoFromHuman first. This ensures you have the necessary details for accurate results.
                """

# Global function to create a tool response after ai messages
def create_tool_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

# Functions to set up logger to build memory in both experience and info
# remember to add current time
# what ever question is asked, the agent should somehow refer to the memory
# For info, we could do RAG, for experience, we should do full-context learning but only for related experinece
def setup_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
        
    handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def setup_agent_logger(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    log_file_path = os.path.join(folder_path, 'agent.log')
    return setup_logger('agent_logger', log_file_path)

def setup_conversation_logger(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    log_file_path = os.path.join(folder_path, 'conversation.log')
    return setup_logger('conversation_logger', log_file_path)

class UserExitError(Exception):
    def __init__(self, message):
        self.message = message

class RequestAdditionalInfoFromHuman(BaseModel):
    """
    Relay the request in a certain format
    Become one of the tools in the tool node, critical for the agent to ask for additional information
    """
    request: str

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # only messages are appended
    supervisor_mode: bool
    query_human: bool

class ToolNode:
    """
    A node that runs the tool calls requested in the last AI message
    An integration of checking if there is a tool call but also running the tool
    There could be more than one tool call in the last AI message. 

    We'll try to use single tool node structure to avoid architecture complications, and also to keep
    the flexibility of adding and deleting tools
    """
    def __init__(self, tools:list):
        self.tools_by_name = {t.name: t for t in tools} # create a dictionary of tools

    def __call__(self, inputs:dict):
        if messages := inputs.get("messages", []): # return an empty as the default value 
            message = messages[-1]
        else:
            raise ValueError("No messages in the input")
       
        # iterate through the tool calls in the message
        # multiple tools calls should result in multiple outputs
        outputs = []
        for tool_call in message.tool_calls:
            
            tool_result = self.tools_by_name[tool_call['name']].invoke(tool_call['args'])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class AgentM:
    def __init__(self, model, tools):
        self.model = model.bind_tools(tools + [RequestAdditionalInfoFromHuman]) # add the new tool to the model
        self.tools = {t.name: t for t in tools}

        # set up tool node
        tool_node = ToolNode(tools=tools)

        # set up memory for multi-run interaction (in memory)
        memory = SqliteSaver.from_conn_string(":memory:")

        # set up the graph
        graph = StateGraph(AgentState) 
        graph.set_entry_point("llm")
        graph.add_node("llm", self.call_llm)
        graph.add_node("tools", tool_node) 
        graph.add_node("query", self.query_node)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_edge("tools", "llm") 
        graph.add_edge("query", "llm")
        graph.add_edge("supervisor", "llm")
        graph.add_conditional_edges(
            "llm",
            self.select_next_node_after_llm,
            {"query": "query", "supervisor":"supervisor" ,"tools": "tools", "__end__": END}
        )
        self.graph = graph.compile(
            checkpointer=memory) # the langchain interrupt must be used with a checkpointer??? for a session?
            # interrupt_before=["tools"],) # adds human oversight to the tool node
            # Note: can also interrupt __after__ actions, if desired.
            # interrupt_after=["tools"]
    
    def call_llm(self, state: AgentState):
        messages = state['messages']
        response = self.model.invoke(messages) 
        query_human = False
        # let the llm decide if we need to request user for additional information
        if len(response.tool_calls) > 0 and response.tool_calls[0]['name'] == RequestAdditionalInfoFromHuman.__name__:
            query_human = True
        return {"messages": [response], "query_human": query_human}

    
    def tool_calls_requested(self, state: AgentState):
        last_message = state['messages'][-1]
        return len(last_message.tool_calls) > 0

    # Integrate the supervisor mode into the first condition
    def select_next_node_after_llm(self, state: AgentState):
        last_ai_message = state["messages"][-1]
        if len(last_ai_message.tool_calls) > 0:
            if state["query_human"]:
                return "query"
            elif state["supervisor_mode"]:
                last_ai_message = state["messages"][-1]
                if last_ai_message.tool_calls[0]['name'] != RequestAdditionalInfoFromHuman.__name__:
                    print("Tool calls to be executed:")
                    for tool_call in last_ai_message.tool_calls:
                        print(f"Action: {tool_call['name']}; Query: {tool_call['args']['query']}")
                    while True:
                        user_input = input("Do you approve the next steps? (y/n/exit) \n").strip().lower()
                        if user_input == 'exit':
                            raise UserExitError("User has exited")
                        elif user_input == "y":
                            return "tools"
                        elif user_input == "n":
                            return "supervisor"
                        else:
                            print("Invalid input. Please enter 'y', 'exit', or 'n'.")
                else: 
                    raise ValueError("Should go to the human node instead of the supervisor node")
            else:
                return "tools"
        else:
            # the evaluator will be invoked when the __end__ is returned
            # return "__end__"
            if state["supervisor_mode"]:
                # we need to display the last ai message and ask for confirmation
                print(last_ai_message.content)
                while True:
                    user_input = input("Are you satified with the solution? (y/n/exit) \n").strip().lower()
                    if user_input == 'exit':
                        raise UserExitError("User has exited")
                    elif user_input == "y":
                        return "__end__"
                    elif user_input == "n":
                        return "supervisor"
                    else:
                        print("Invalid input. Please enter 'y', 'exit', or 'n'.")
            else:
                print(last_ai_message.content)
                return "__end__"
    


    def query_node(self, state: AgentState):
        """
        Human node anwsers the query from the agent and also provides feedback under supervisor mode.
        """
        # we can tell if the query_node is for supervisor mode or for additional information by checking the status of query_human
        # We have to provide information not during interrupt but during the node execution, so the previous message should be an AI message.
        
        
        last_ai_message = state["messages"][-1]
        print(last_ai_message)
        if state["query_human"]:
            request = last_ai_message.tool_calls[0]['args']['request']
            user_input = input(f"[Agent is requiring additional information] {request} \n")
            if user_input.lower() == 'exit':
                raise UserExitError("User has exited")
            else:
                # An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'.
                new_human_tool_message = create_tool_response(user_input, last_ai_message)
                # remember to unset the query_human flag
                return {"messages": [new_human_tool_message], "query_human": False}


    def supervisor_node(self, state: AgentState):
        """
        Supervisor mode, where the human can provide feedback on the agent's actions or confirm the next action
        The comment should imply that we make a mistake in the past, we should go back and manually adjust the state. eg. reframe the qeury or re-think the action.
        Sumply adding human comment as a tool message to the ai message expecting a search result is not enough. 
        Can we also get rid of the ai message that contains the original tool call?

        # make sure the supervisor_mode is True
        # make sure the next action is a non-query tool call
        # if the next action is not a query tool call, then the human should be asked for confirmation and then go to the tool node
        # For multiple tool calls, the human should approve all tool calls at once, or leave feedback and re-think
        # if the action cannot be confirmed, the human should provide feedback and then go back to the llm node
        # Otherwise, 'exit' will end the entire workflow 
        """
        
        existing_ai_message = state["messages"][-1]
        
        # get human comment
        comment = input("[Type your feedback below] \n")
        if comment.lower() == 'exit':
            raise UserExitError("User has exited")
        else:
            new_human_message = HumanMessage(
                content=f"{comment} \n {SYSTEM_PROMPT}",
                # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
                id=existing_ai_message.id,
            )
            return {"messages": [new_human_message]}
        

    def evaluator(self, state: AgentState):
        """
        Works only while the supervisor mode is on. 
        Every time the conditional edge make a __end__ decision, the evaluator will be called.
        Use the (y/n/exit) to decide the next action.

        Consider the time travel function. 

        options:
        if y, go to the END
        if exit, raise UserExitError
        if n, go back to the supervisor node
        """
        pass


        
def main():

    # add args
    parser = argparse.ArgumentParser()
    parser.add_argument("supervisor_mode_on", action="store_true", help="Turn on the supervisor mode")
    args = parser.parse_args()

    # prepare for model
    model = ChatOpenAI(model="gpt-4o")

    # prepare for tool node, replace with all web actions
    tool = TavilySearchResults(max_results=3)
    tools = [tool]

    # create the agent 
    # system_prompt = "You are an AI agent and can answer people's questions.For any interaction with humans, you can ask for additional information using the tool 'RequestAdditionalInfoFromHuman', which should be used ahead of any tool calls."
    # system_prompt = "Don't speak antyhing."
    agent = AgentM(model, tools)

    # the initial task input
    # messages = [HumanMessage(content="What is score of NBA Final Game 1? \n {system_prompt}")]
    # messages = [HumanMessage(content=f"I want to know the nba score. \n {SYSTEM_PROMPT}")]
    messages = [HumanMessage(content=f"What's the age of the president? \n {SYSTEM_PROMPT}")]
    # messages = [HumanMessage(content="What is nba score last night? \n {system_prompt}")]
    # messages = [HumanMessage(content=f"The total number of students in my class is to use my student id and plus 10. So how many students are there in my class?")]
    # messages = [HumanMessage(content=f"The total number of students in my class is to use my student id and plus 10. So how many students are there in my class? \n {system_prompt}")]
    # messages = [SystemMessage(content=system_prompt)] + messages
    # print(messages) 

    # Execute the task
    try: 
        result = agent.graph.invoke({"messages": messages, "supervisor_mode": args.supervisor_mode_on}, config=config)
        # result = agent.graph.invoke({"messages": messages})
        # print(result['messages'][-1].content)
        # print("*" * 20)
        # print(agent.graph.get_state(config))
    except UserExitError as e:
        print(e.message)
    # except Exception as e:
    #     print(e)



if __name__ == "__main__":
    main()