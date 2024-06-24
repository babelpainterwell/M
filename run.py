from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.pydantic_v1 import BaseModel
import json
import argparse
from prompts import SYSTEM_PROMPT, TOOL_PROMPT
from utils import mark_page, conclude_step, conclude_info
from tools import click, type_text, scroll, wait, go_back, to_google, upload, finish
from state import AgentState
import asyncio
from playwright.async_api import async_playwright
from langchain_core.prompt_values import PromptValue
from typing import List
from store_memory import create_new_experience, add_step, add_info
from retrieve_memory import retrieve_info_memory, retrieve_experience_memory
from utils import Data, Info
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA


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


# Mmeory Path
experiences_path = "memory/experiences.json"
info_path = "memory/info.json"



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







# function to annotate the page 
async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page} # update the state

def format_bbox_observation(state: AgentState) -> str:
    """
    Describe the bounding boxes in the state, only after the state["bounding_box"] has been updated
    """
    labels = []
    for i, bbox in enumerate(state["bounding_boxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_observation = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return bbox_observation




# annotate the page -> form the observation -> feed to the agent
# we should keep a list observations
# observations is a list of JSON objects
"""
Format:
{
    "bbox_observation": "The bounding boxes are ...",
    "action": "Click",
    "args": [5],
    "human_feedback": "The bounding box is not correct"
}
"""

def parse(text: str) -> dict:
    action_prefix = "Action: "
    summary_prefix = "Summary: "
    thought_prefix = "Thought: "

    lines = text.strip().split("\n")

    # Check if the last line starts with the action prefix
    if not lines[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}", "summary": None, "thought": None}

    # Extract the action block
    action_block = lines[-1]
    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [inp.strip().strip("[]") for inp in action_input.strip().split(";")]

    # Extract the summary
    summary = None
    for line in lines:
        if line.startswith(summary_prefix):
            summary = line[len(summary_prefix):].strip()
            break
    
    # Extract the thought
    thought = None
    for line in lines:
        if line.startswith(thought_prefix):
            thought = line[len(thought_prefix):].strip()
            break

    if action not in ["Click", "Type_Text", "Scroll", "Wait", "Go_Back", "To_Google", "Upload", "FINISH"]:
        print(f"********************* Invalid action: {action} ********************")
        return {"action": "retry", "args": f"Invalid action: {action}", "summary": None, "thought": None}

    print("the action is: ", action)
    return {"action": action, "args": action_input, "summary": summary, "thought": thought}


# Update the observations automatically by returning an updated state




class UserExitError(Exception):
    def __init__(self, message):
        self.message = message

class RequestAdditionalInfoFromHuman(BaseModel):
    """
    Relay the request in a certain format
    Become one of the tools in the tool node, critical for the agent to ask for additional information
    """
    request: str

# class ToolNode:
#     """
#     A node that executes the action in the last AI message
#     An integration of checking if there is a tool call but also running the tool
    
#     One tool node corresponds to one tool call
#     """
#     def __init__(self, tool_name, tool):
#         self.tool_by_name = {tool_name: tool}

#     def __call__(self, inputs:dict):
#         if messages := inputs.get("messages", []): # return an empty as the default value 
#             message = messages[-1]
#         else:
#             raise ValueError("No messages in the input")
       
#         # iterate through the tool calls in the message
#         # multiple tools calls should result in multiple outputs
#         outputs = []
#         the_tool_to_run_name = message.tool_calls[0]['name']
#         if the_tool_to_run_name not in self.tool_by_name:
#             raise ValueError(f"Tool {the_tool_to_run_name} not found")
#         else:
#             tool_result = self.tool_by_name[the_tool_to_run_name].invoke(message.tool_calls[0]['args'])
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=the_tool_to_run_name,
#                     tool_call_id=message.tool_calls[0]["id"],
#                 )
#             )
#         return {"messages": outputs} # then how to update the observations?  doesn't have to be tool node 



class AgentM:
    def __init__(self, model, tools: dict, image_summarizer_model):
        self.model = model.bind_tools([RequestAdditionalInfoFromHuman]) # add the new tool to the model
        self.tools = tools
        extractor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert extraction algorithm. "
                    "Only extract relevant personal information from the text including account credentials, personal preferences or background."
                    "Extracted infomation should be specific and detailed eg. 'the user's name for a Zillow job application' or 'user's email address used to register for LinkedIn'."
                    "If you do not know the value of an attribute asked to extract, "
                    "return null for the attribute's value.",
                ),
                # Please see the how-to about improving performance with
                # reference examples.
                # MessagesPlaceholder('examples'),
                ("human", "{text}"),
            ]
        )
        self.info_extractor_chain = extractor_prompt | model.with_structured_output(schema=Info)
        self.image_summarizer_model = image_summarizer_model

        # set up tool nodes
        # tool_nodes = {name: ToolNode(name, tool) for name, tool in self.tools.items()}
        
        # set up memory for multi-run interaction (in memory)
        # memory = SqliteSaver.from_conn_string(":memory:")

        # define the llm/agent node 

        # set up the graph
        graph = StateGraph(AgentState) 
        graph.set_entry_point("update_observations")
        graph.add_node("update_observations", self.update_observations)
        graph.add_edge("update_observations", "llm")
        graph.add_node("llm", self.call_llm)
        graph.add_node("query", self.query_node)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_edge("query", "update_observations")
        graph.add_edge("supervisor", "update_observations")

        
        # add tool nodes
        # eg. name: 'Click ', node: ToolNode('Click', click)
        for name, func in tools.items():
            graph.add_node(name, func)
            if name != "Finish":
                graph.add_edge(name, "update_observations")

        # end the graph with the finish node
        graph.add_edge("Finish", END)
        

        graph.add_conditional_edges(
            "llm",
            self.select_next_node_after_llm,
            {"query": "query", "supervisor":"supervisor" , "__end__": END,
             "click" : "Click", "type_text": "Type_Text", "scroll": "Scroll", 
             "wait": "Wait", "go_back": "Go_Back", "to_google": "To_Google", 
             "upload": "Upload", "finish": "Finish", "llm": "llm"}
        )
        self.graph = graph.compile()
            # checkpointer=memory) # the langchain interrupt must be used with a checkpointer??? for a session?
            # interrupt_before=["tools"],) # adds human oversight to the tool node
            # Note: can also interrupt __after__ actions, if desired.
            # interrupt_after=["tools"]

    
    def call_llm(self, state: AgentState):
        # print("*" * len("*Visiting the llm node*"))
        # print("*Visiting the llm node*")
        # print("*" * len("*Visiting the llm node*"))
        print("Agent is reasoning...")
        

        # Extract and print messages
        messages = state.get('messages', [])

        # Ensure messages are of an expected type
        if not isinstance(messages, (PromptValue, str, list)):
            raise TypeError("messages must be a PromptValue, str, or list of BaseMessages")

        # Invoke the model
        response = self.model.invoke(messages)

        query_human = False
        # print(response)

        # Let the LLM decide if we need to request user for additional information
        if len(response.tool_calls) > 0:
            print("**********************Tool Calls Detected *******************")
            # check if the tool call is 'RequestAdditionalInfoFromHuman'
            if response.tool_calls[0].get('name') == RequestAdditionalInfoFromHuman.__name__:
                query_human = True
                print("**********************QUERY CALL *******************")
            return {**state, "messages": [response], "query_human": query_human} # if go to the query node, there is no prediciton 
        else:
            # update the prediction to the state
            result = parse(response.content)
            # if requestAdditionalInfoFromHuman in the action, then go to the query node
            if result['action'].startswith("RequestAdditionalInfoFromHuman"):
                query_human = True
                print("**********************QUERY CALL (The action starts with RequestAdditionalInfoFromHuman) *******************")
                return {**state, "messages": [response], "query_human": query_human}

            # prediction is added to agentState but not into the observation yet.
            pred = {"action": result["action"], "args": result["args"], "summary": result["summary"], "thought": result["thought"]}
            return {**state, "messages": [response], "prediction": pred, "query_human": query_human}

    
    def tool_calls_requested(self, state: AgentState):
        last_message = state['messages'][-1]
        return len(last_message.tool_calls) > 0


    def select_next_node_after_llm(self, state: AgentState):
        # last_ai_message = state["messages"][-1]
        # print("LAST AI MESSAGE IS: ", last_ai_message)
        # if len(last_ai_message.tool_calls) > 0: # now tool call can only be 'query'
        if state["query_human"]: # if the action is a query, then go to the query node without asking for confirmation
            print("**********************Go To Query Without Confirmation *******************")
            return "query"
        else:
            """
            The END condition should be once the agent has finished its task, meaning return answer 
            Our current ending condition is whenever there is no tool call in the last ai message
            """
            prediction = state["prediction"]
            if prediction['action'] == "retry":
                print(prediction['args'])
                return "llm"

            print(f"Bounding Box Description: \n {state['observations'][-1]['bbox_observation']}")
            print(f'[VISUAL SUMMARY] {prediction["summary"]}')
            print(f'[THOUGHT] {prediction["thought"]}')
            print(f"Action: {prediction['action']}, Args: {prediction['args']}")
            print("*" * 50)
            if state["supervisor_mode"]:
                if "RequestAdditionalInfoFromHuman" in prediction['action']: 
                    print("**********************Query While Tool Call Is Zero *******************")
                    return "query"
                if prediction['action'] == "FINISH":
                    while True:
                        user_input = input("Are you satified with the solution? (y/n/exit) \n").strip().lower()
                        if user_input == 'exit':
                            raise UserExitError("User has exited")
                        elif user_input == "y":
                            # add to memory start
                            step = conclude_step(state)
                            add_step(experiences_path, step)
                            # add to memory end
                            return "__end__"
                        elif user_input == "n":
                            return "supervisor"
                        else:
                            print("Invalid input. Please enter 'y', 'exit', or 'n'.")
                else:
                    while True:
                        user_input = input("Do you approve the next steps? (y/n/exit) \n").strip().lower()
                        if user_input == 'exit':
                            raise UserExitError("User has exited")
                        elif user_input == "y":
                            # add to memory start
                            step = conclude_step(state)
                            add_step(experiences_path, step)
                            # add to memory end
                            if prediction['action'] == "Click":
                                return "click"
                            elif prediction['action'] == "Type_Text":
                                return "type_text"
                            elif prediction['action'] == "Scroll":
                                return "scroll"
                            elif prediction['action'] == "Wait":
                                return "wait"
                            elif prediction['action'] == "Go_Back":
                                return "go_back"
                            elif prediction['action'] == "To_Google":
                                return "to_google"
                            elif prediction['action'] == "Upload":
                                return "upload"
                        elif user_input == "n":
                            return "supervisor"
                        else:
                            print("Invalid input. Please enter 'y', 'exit', or 'n'.")

            else:
                if prediction['action'] == "Click":
                    return "click"
                elif prediction['action'] == "Type_Text":
                    return "type_text"
                elif prediction['action'] == "Scroll":
                    return "scroll"
                elif prediction['action'] == "Wait":
                    return "wait"
                elif prediction['action'] == "Go_Back":
                    return "go_back"
                elif prediction['action'] == "To_Google":
                    return "to_google"
                elif prediction['action'] == "Upload":
                    return "upload"
                elif prediction['action'] == "FINISH":
                    return "finish"
                else:
                    raise ValueError(f"Invalid action {prediction['action']}")
    


    def query_node(self, state: AgentState):
        """
        Human node answers the query from the agent and also provides feedback under supervisor mode.

        We also fetch info from memory and write into info memory.
        """
        # print("*" * len("*Visiting the query node*"))
        # print("*Visiting the query node*")
        # print("*" * len("*Visiting the query node*"))
        
        last_ai_message = state["messages"][-1]
        if state["query_human"]:
            try:
                request = last_ai_message.tool_calls[0]['args']['request']
            except (KeyError, IndexError) as e:
                raise ValueError("Invalid structure in last AI message for tool_calls") from e

            try:
                memory_result = retrieve_info_memory(request)  # Attempt to retrieve information from memory
                memory_result_str = str(memory_result[0].page_content)
                print(f"Memory Fetched: {memory_result_str}")
                new_tool_message = create_tool_response(memory_result_str, last_ai_message)
                return {"messages": [new_tool_message], "query_human": False}
            except Exception as e:
                print(f"[Agent could not find the information in memory]")

            while True:
                user_input = input(f"[Agent is requiring additional information] {request} \n")
                if user_input.lower() == 'exit':
                    raise UserExitError("User has exited")

                if user_input.strip():
                    new_human_tool_message = create_tool_response(user_input, last_ai_message)
                   
                    # add to memory start
                    text = f"Request: {request} \n Response: {user_input}"
                    info = self.info_extractor_chain.invoke({"text": text})
                    if info.info:
                        info_dict = conclude_info(info)
                        if info_dict["info"]:
                            add_info(info_path, info_dict)
                    # add to memory end

                    return {"messages": [new_human_tool_message], "query_human": False}
                else:
                    print("Invalid input. Please enter your response or type 'exit' to terminate.")
                    # print(last_ai_message)
        else:
            raise ValueError("Query node should only be executed when query_human is True.")


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
        # print("*" * len("*Visiting the supervisor node*"))
        # print("*Visiting the supervisor node*")
        # print("*" * len("*Visiting the supervisor node*"))
        
        existing_ai_message = state["messages"][-1]
        
        # get human comment
        while True:
            comment = input("[Type your feedback below] \n")
            
            if comment.lower() == 'exit':
                raise UserExitError("User has exited")
            
            if comment.strip():
                new_human_message = HumanMessage(
                    content=f"{comment}",
                    id=existing_ai_message.id,  # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
                )

                # add to memory start
                step = conclude_step(state, comment)
                add_step(experiences_path, step)
                # add to memory end

                # add to memory start
                text = f'Thought: {state["prediction"]["thought"]}; Comment: {comment}'
                info = self.info_extractor_chain.invoke({"text": comment})
                if info.info:
                    info_dict = conclude_info(info)
                    if info_dict["info"]:
                        add_info(info_path, info_dict)
                # add to memory end

                # Update the state with the human feedback
                observation = state.get("observations", [{}])[-1]
                if observation["human_feedback"] is not None:
                    observation["human_feedback"].append(comment)
                else:
                    raise ValueError("Human feedback is None; should be an empty list at least.")
                
                state["observations"][-1] = observation

                # add the human feedback to memory 
                # try: 
                #     summary = state["prediction"]["summary"]
                #     create_new_query(summary, comment) 
                # except Exception as e:
                #     print(e)
                #     print("Could not add the human feedback to memory")

                # Return the updated state with the new human message
                return {**state, "messages": [new_human_message]}
            else:
                print("Invalid input. Please enter your feedback or type 'exit' to terminate.")
        
    
    async def update_observations(self, state: AgentState):
        """
        Annotate the latest screenshot and update the observations
        Only place where an Observation is added to the state
        """
        # print("*" * len("*Visiting the update_observations node*"))
        # print("*Visiting the update_observations node*")
        # print("*" * len("*Visiting the update_observations node*"))
        print("Updating the observations...")


        # Clip the message with image so that we don't keep the image in the state
        # Also need to clip the message with mempry
        # Otherwise, the state will be too large and the cost is high 
        try: 
            if len(state["messages"]) > 1:
                # look for the message which contains the image
                for i in range(len(state["messages"])):
                    if isinstance(state["messages"][i], HumanMessage): # human feedback is also Human message, the content is just a string
                        # check the if content not a string
                        if type(state["messages"][i].content) == list: # contains image or memory or both 
                            if len(state["messages"][i].content) > 1: # note that there could be memory in the content
                                if state["messages"][i].content[1].get("type") == "image_url":
                                    state["messages"][i].content = state["messages"][i].content[:1]
                                    state["messages"][i].content[0]["text"] = "The screenshot has been removed for security and efficiency reasons. But the screenshot is still available in the latest observation in each iteration."
                                    print("PREVIOUS IMAGE HAS BEEN FOUND AND CLIPEED")
                                    break


            # remove entries where the content attribute of a HumanMessage is a list, especially those that contain an image or memory.
            # state['messages'] = [msg for msg in state['messages'] if not (isinstance(msg, HumanMessage) and isinstance(msg.content, list))]
        except Exception as e:
            print(e)
            # print(state["messages"])
        
        state = await annotate(state)  # the updated part was not updated to the main state 
        
        bbox_observation = format_bbox_observation(state)
        # print(bbox_observation)

        b64_img = state["img"]
        
        text_content = (
            "Observation: please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n"
            f"{bbox_observation}"
            f"{TOOL_PROMPT}"
        )


        image_summary_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ]
        )

        summary_messages = [SystemMessage(content="You are an image summarizer, tasked with providing as detailed summary as possible of the image. Summarize what you see from the screenshot and it needs to be as detailed as possible; such as the content of the webpage, the position of the elements, etc. Pay extra attention to the details such as if input boxes have been filled or not, etc."), image_summary_message]

        image_summary = self.model.invoke(summary_messages).content
        # print(f"Image Summary: {image_summary}")

        
        # Updating the observations; note that in the beginning, the observations is an empty list
        updated_observations = state.get("observations", [])
        if updated_observations:
            updated_observations.append({"bbox_observation": bbox_observation, "image_summary": image_summary, "human_feedback": []}) # human_feedback is initialized at the very beginning
        else:
            updated_observations = [{"bbox_observation": bbox_observation,"image_summary": image_summary, "human_feedback": []}]

        try: 
            memory = retrieve_experience_memory(bbox_observation, image_summary)

            memory_content = (
            "Memory: Below are previous situations similar to the current context, which include descriptions of the situations, actions taken, and the reasoning behind those actions, as well as feedback on whether the actions were reasonable. You can refer to these contexts, which may potentially help you formulate your thoughts and actions now, though not always.\n"
            f"{memory}"
            )

            new_image_message = HumanMessage(
                content=[
                    {"type": "text", "text": text_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                    {"type": "text", "text": memory_content},
                ]
            )
        except Exception as e:
            print("Could not retrieve memory")
            new_image_message = HumanMessage(
                content=[
                    {"type": "text", "text": text_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                ]
            )
        
        # Return the updated state
        return {**state, "messages": [new_image_message], "observations": updated_observations}
    

        
async def main():

    # add args
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor_mode_on", action="store_true", help="Turn on the supervisor mode")
    parser.add_argument("--supervisor_mode_off", action="store_false", dest="supervisor_mode_on", help="Turn off the supervisor mode")
    args = parser.parse_args()

    print(f"Starting the agent with supervisor mode {'on' if args.supervisor_mode_on else 'off'}")
    

    # prepare for model
    model = ChatOpenAI(model="gpt-4o")
    # image_summrizor_model = ChatOpenAI(model="gpt-4o")
    llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

    # prepare for tool node, replace with all web actions
    tools = {
        "Click": click,
        "Type_Text": type_text,
        "Scroll": scroll,
        "Wait": wait,
        "Go_Back": go_back,
        "To_Google": to_google,
        "Upload": upload,
        "Finish": finish
    }

    # create the agent 
    agent = AgentM(model, tools, llm)

    # Start in full screen mode
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )
        context = await browser.new_context()
        page = await context.new_page()

        
        # target_url = "https://zillow.wd5.myworkdayjobs.com/Zillow_Group_External/job/Remote-USA/Machine-Learning-Engineer_P744869-1?source=LinkedIn"
        # target_url = "https://zillow.wd5.myworkdayjobs.com/en-US/Zillow_Group_External/job/Remote-USA/Machine-Learning-Engineer_P744869-1/apply/autofillWithResume?source=LinkedIn"
        # target_url = "https://careers.docusign.com/jobs/24746?lang=en-us&iis=Job+board&iisn=LinkedIn"
        target_url = "https://wd1.myworkdaysite.com/en-US/recruiting/snapchat/snap/job/Bellevue-Washington/Software-Engineer--ML-Infrastructure-3--Years-of-Experience_R0035754-2?source=LinkedIn"
        await page.goto(target_url)
        await asyncio.sleep(5) # wait for the page to load

        create_new_experience() # create a new experience in the memory
        try:
            while True:
                state = {
                    "messages": [SystemMessage(content=SYSTEM_PROMPT)], 
                    "supervisor_mode": args.supervisor_mode_on, 
                    "query_human": False, 
                    "page": page, 
                    "target_url": target_url
                }
                result = await agent.graph.ainvoke(state, {"recursion_limit": 1500})
                if result == END:
                    break
        except UserExitError as e:
            print(e.message)

        # try to catch different exceptions and it's best if we can continue from we are left. 

if __name__ == "__main__":
    asyncio.run(main())