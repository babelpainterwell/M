import base64
from langchain_core.runnables import chain as chain_decorator
import asyncio
import os
from state import AgentState
from datetime import datetime
from state import Step, InfoDict, DataDict
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional



class Data(BaseModel):
    """Personal information about the user's account credentials, preferneces, or background presented in a Q&A format, consisting of a question and its answer. The question should be specific; for example, if it concerns an email address, it should specify what the email is used for, such as 'the email used to register a job application account for Zillow.'"""
    ques: Optional[str] = Field(default=None, description="The question regarding the user's information, which should be specific, e.g., 'the user's name for a Zillow job application' or 'user's email address used to register for LinkedIn'")
    ans: Optional[str] = Field(default=None, description='The answer to the question')



class Info(BaseModel):
    """ONLY personal information about the user, consisting of a detailed list of multiple personal data points. It needs to be specific."""
    info: List[Data] = Field(description="A list of personal information about the user's account credentials, preferneces, or background. ")


@chain_decorator
async def mark_page(page):
    """
    return a screenshot of the page and the bounding boxes
    """
    # Read the JavaScript file
    with open("mark_page.js") as f:
        mark_page_script = f.read()
    
    # Evaluate the script in the page context
    await page.evaluate(mark_page_script)
    
    # Retry mechanism to ensure the page is fully loaded
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            await asyncio.sleep(3)  # Wait for more time to load the page if initial trials fail
    
    # Take a screenshot of the page
    await asyncio.sleep(5) # more time to take screenshot
    screenshot = await page.screenshot()

    # save the screenshot to test_screenshots folder 
    index = len(os.listdir("test_screenshots"))
    with open(f"test_screenshots/screenshot_{index}.png", "wb") as f:
        f.write(screenshot)
    
    # Clean up by removing the annotations
    await page.evaluate("unmarkPage()")
    
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bounding_boxes": bboxes,
    }
    

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    


def conclude_step(state: AgentState, additional_feedback: str = "") -> Step:
    try:
        observation = state["observations"][-1]
        bbox_description = observation["bbox_observation"]
        visual_summary = observation["image_summary"]
        context_summary = {
            "bbox_description": bbox_description,
            "visual_summary": visual_summary,
        }
        
        # Safely generate image_id
        screenshot_files = os.listdir('test_screenshots')
        if not screenshot_files:
            raise FileNotFoundError("No screenshots found in the directory.")
        
        image_id = f"test_screenshots/screenshot_{len(screenshot_files) - 1}.png"
        
        agent_thought = state["prediction"]["thought"]
        agent_action = state["prediction"]["action"]
        
        # Construct human feedback
        base_feedback = "Agent action is correct." if not additional_feedback else "Agent action needs improvement. Feedback to the current action:"
        human_feedback = base_feedback + " " + additional_feedback.strip()
        
        time = datetime.now().isoformat()

        step: Step = {
            "context_summary": context_summary,
            "image_id": image_id,
            "agent_thought": agent_thought,
            "agent_action": agent_action,
            "human_feedback": human_feedback,
            "timestamp": time,
        }
        return step
    except KeyError as e:
        raise ValueError(f"Missing key in state or observations: {e}")
    except FileNotFoundError as e:
        raise



def conclude_info(info: Info) -> InfoDict:
    # Convert Info object to InfoDict format
    info_list = [{'ques': d.ques, 'ans': d.ans} for d in info.info if d.ques and d.ans]
    timestamp = datetime.now().isoformat()
    return InfoDict(info=info_list, timestamp=timestamp)
 




    
    
