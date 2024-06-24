from typing import List, Annotated, Any
from typing_extensions import TypedDict
from typing import Optional
from playwright.async_api import Page
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# Define web action elements
class BoundingBox(TypedDict):
    """
    ariaLabel is used to enhance the accessibility of web applications
    eg. The button contains only an icon (✖) and does not have any visible text.
    The aria-label="Close the dialog" attribute provides a textual label for the button
    """
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    """
    action: Click 
    args: ["5"]
    """
    action: str 
    args: Optional[List[str]] # note it should be a list of strings
    summary: str
    thought: str

class Observation(TypedDict):

    bbox_observation: str
    image_summary: str
    prediction: Prediction
    human_feedback: List[str]
    timestamp: str

# class Task(TypedDict):
#     task_id: str
#     observations: List[Observation]

class ContextSummary(TypedDict):
    bbox_description: str
    visual_summary: str

class Step(TypedDict):
    context_summary: ContextSummary
    image_id: str 
    agent_thought: str
    agent_action: str
    human_feedback: str
    timestamp: str

class Experience(TypedDict):
    experience_id: str
    steps: List[Step]

class DataDict(TypedDict):
    ques: str
    ans: str

class InfoDict(TypedDict):
    info: List[DataDict]
    timestamp: str
    


class AgentState(TypedDict):
    """
    Feedback Loop: Observation -> Thought -> Action -> Observation
    """
    messages: Annotated[list[AnyMessage], add_messages] 
    supervisor_mode: bool
    query_human: bool
    page: Page 
    target_url: str 
    img: str # b64 encoded screenshot 
    bounding_boxes: List[BoundingBox]
    observations: List[Observation]
    prediction: Prediction

    # Note that excessive observation of web pages from longer episodes can confuse the agent