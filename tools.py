from state import AgentState
import platform
import asyncio
import os


"""
They should update the observations, at the end of each tool, they should update the img. 
Prediction and human feedback is updated at the end of each tool call. 

update the observations! The core is to make sure the page has been updated


HUMAN_FEEDBACK is updated at the supervisor node.
"""


async def click(state: AgentState):
    """
    update page 
    update img, updated at the update_obeservation node 
    bbox_observation also updated at the update_observation node
    update observations, in which Prediction and human feedback should be updated
    """
    # print("*"*20)
    # print("Visiting click node")
    # the prediction hasb't been updated yet, not the prediction in observation
    print("Agent is taking action...")
    page = state.get("page")
    click_args = state.get("prediction", {}).get("args")
    print(f"click_args: {click_args}")
    print("*"*20) 
    

    if not click_args or len(click_args) != 1:
        error_msg = f"Failed to click bounding box labeled as number {click_args}"
        print(error_msg)
        return {**state}

    try:
        bbox_id = int(click_args[0])
        bbox = state["bounding_boxes"][bbox_id]
    except (ValueError, IndexError, KeyError) as e:
        error_msg = f"Error accessing bounding box {bbox_id}: {str(e)}"
        print(error_msg)
        return {**state}

    x, y = bbox["x"], bbox["y"]

    # Ensure focus is on the webpage by focusing the body element
    try:
        await page.evaluate("document.body.focus();")
        print("Focused on the body element to ensure focus on the webpage")
    except Exception as e:
        error_msg = f"Error focusing on the body element: {str(e)}"
        print(error_msg)
        return {**state}

    try:
        await page.mouse.click(x, y)
        print(f"Clicked bounding box {bbox_id} at ({x}, {y})")
    except Exception as e:
        error_msg = f"Error clicking at ({x}, {y}): {str(e)}"
        print(error_msg)
        return {**state}

    # Update the observation prediction
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}



async def type_text(state: AgentState):
    print("Agent is taking action...")
    page = state.get("page")
    type_args = state.get("prediction", {}).get("args")
    
    if not type_args or len(type_args) != 2:
        error_msg = f"Failed to type in element from bounding box labeled as number {type_args}"
        print(error_msg)
        return {**state}
    
    # Ensure focus is on the webpage by focusing the body element
    try:
        await page.evaluate("document.body.focus();")
        print("Focused on the body element to ensure focus on the webpage")
    except Exception as e:
        error_msg = f"Error focusing on the body element: {str(e)}"
        print(error_msg)
        return {**state}

    try:
        bbox_id = int(type_args[0])
        bbox = state["bounding_boxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        text_content = type_args[1]

        await page.mouse.click(x, y)
        
        # Check if MacOS and clear all text before entering new
        select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
        await page.keyboard.press(select_all)
        await page.keyboard.press("Backspace")
        await page.keyboard.type(text_content)
        await page.keyboard.press("Enter")
        
        print(f"Typed {text_content} and submitted")
    except Exception as e:
        error_msg = f"Error typing text: {str(e)}"
        print(error_msg)
        return {**state}

    # Update the observation
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}

async def scroll(state: AgentState):
    print("Agent is taking action...")
    page = state.get("page")
    scroll_args = state.get("prediction", {}).get("args")
    
    if not scroll_args or len(scroll_args) != 2:
        error_msg = "Failed to scroll due to incorrect arguments."
        print(error_msg)
        return {**state}

    target, direction = scroll_args

    # Ensure focus is on the webpage by focusing the body element
    try:
        await page.evaluate("document.body.focus();")
        print("Focused on the body element to ensure focus on the webpage")
    except Exception as e:
        error_msg = f"Error focusing on the body element: {str(e)}"
        print(error_msg)
        return {**state}

    try:
        if target.upper() == "WINDOW":
            print("Scrolling in window")
            scroll_amount = 500
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        else:
            print("Scrolling in element")
            scroll_amount = 200
            target_id = int(target)
            bbox = state["bounding_boxes"][target_id]
            x, y = bbox["x"], bbox["y"]
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)

        print(f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}")
    except Exception as e:
        error_msg = f"Error scrolling: {str(e)}"
        print(error_msg)
        return {**state}

    # Update the observation
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}

async def wait(state: AgentState):
    print("Agent is taking action...")
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    print(f"Waited for {sleep_time}s.")

    # Update the observation
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}

async def go_back(state: AgentState):
    print("Agent is taking action...")
    page = state.get("page")
    try:
        await page.go_back()
        print(f"Navigated back a page to {page.url}.")
    except Exception as e:
        error_msg = f"Error navigating back: {str(e)}"
        print(error_msg)
        return {**state}

    # Update the observation
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}

async def to_google(state: AgentState):
    print("Agent is taking action...")
    page = state.get("page")
    try:
        await page.goto("https://www.google.com/")
        print("Navigated to google.com.")
    except Exception as e:
        error_msg = f"Error navigating to Google: {str(e)}"
        print(error_msg)
        return {**state}
    
    # Update the observation
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}

async def upload(state: AgentState):
    print("Agent is taking action...")
    page = state.get("page")
    
    # The file path to upload is 'resume/CV_Zhongwei.pdf'
    file_path = "resume/CV_Zhongwei.pdf"
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        print(error_msg)
        return {**state}

    try:
        await page.set_input_files('input[type="file"]', file_path)
        print(f"Uploaded file {file_path}.")
    except Exception as e:
        error_msg = f"Error uploading file: {str(e)}"
        print(error_msg)
        return {**state}

    # Update the observation
    observation = state.get("observations", [{}])[-1]
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}

async def finish(state: AgentState):
    # Update the observation
    observation = state.get("observations", [{}])[-1] # the recent observation without the prediction and human feedback
    observation["prediction"] = state.get("prediction")
    state["observations"][-1] = observation

    return {**state}


# class StateAsInput(BaseModel):
#     messages: Annotated[list[AnyMessage], add_messages] = Field(description="The messages exchanged between the user and the agent.")
#     supervisor_mode: bool = Field(description="Whether the agent is in supervisor mode.")
#     query_human: bool = Field(description="Whether the agent should query the human.")
#     # page: Page = Field(description="The page where the agent should perform actions.")
#     input: str = Field(description="The input to the agent.")
#     img: str = Field(description="The screenshot of the page.")
#     bounding_boxes: List[BoundingBox] = Field(description="The bounding boxes on the page.")
#     prediction: Prediction = Field(description="The prediction of the agent.")
#     observation: Annotated[str, add_messages] = Field(description="The observation of the agent.")


# class CustomMarkPageTool(BaseTool):
# class CustomClickTool(BaseTool):
#     name = "click"
#     description = "Click action in the browser - Click [Numerical_Label]"
#     args_schema: Type[BaseModel] = StateAsInput
#     return_direct: bool = True

#     def _run(
#         self, state: AgentState, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         page = state["page"]
#         click_args = state["prediction"]["args"] 
#         if click_args is None or len(click_args) != 1:
#             return f"Failed to click bounding box labeled as number {click_args}"
#         bbox_id = click_args[0]
#         bbox_id = int(bbox_id)
#         try:
#             bbox = state["bboxes"][bbox_id]
#         except Exception:
#             return f"Error: no bbox for : {bbox_id}"
#         x, y = bbox["x"], bbox["y"]
#         page.mouse.click(x, y)
#         return f"Clicked {bbox_id}"


# # class CustomTypeTextTool(BaseTool):
# class CustomTypeTextTool(BaseTool):
#     name = "type_text"
#     description = "Type text in the browser - Type_Text [Numerical_Label, Text]"
#     args_schema: Type[BaseModel] = StateAsInput
#     return_direct: bool = True

#     def _run(
#         self, state: AgentState, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         page = state["page"]
#         type_args = state["prediction"]["args"]
#         if type_args is None or len(type_args) != 2:
#             return f"Failed to type in element from bounding box labeled as number {type_args}"
#         bbox_id = type_args[0]
#         bbox_id = int(bbox_id)
#         bbox = state["bboxes"][bbox_id]
#         x, y = bbox["x"], bbox["y"]
#         text_content = type_args[1]
#         page.mouse.click(x, y)
#         # Check if MacOS and clear all text before entering new
#         select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
#         page.keyboard.press(select_all)
#         page.keyboard.press("Backspace")
#         page.keyboard.type(text_content)
#         page.keyboard.press("Enter")
#         return f"Typed {text_content} and submitted"


# remember to add upload 
# click args: [bbox_id]
# type_text args: [bbox_id, text]
# scroll args: [target, direction]


