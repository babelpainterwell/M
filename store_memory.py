from state import Observation, Experience, ContextSummary, Step, InfoDict
import os
from datetime import datetime
from typing import List
import json


def read_json_file(file_path: str) -> List[Experience]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                # Handle the case where the file is empty or contains invalid JSON
                return []
    return []

def write_json_file(file_path: str, data: List[Experience]) -> None:
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_new_experience() -> None:
    file_path = "memory/experiences.json"
    os.makedirs("memory", exist_ok=True)
    if not os.path.exists(file_path):
        # create the file if it does not exist
        with open(file_path, 'w') as file:
            json.dump([], file)

    data = read_json_file(file_path)

    # new experience_id should be the maximum experience_id + 1
    experience_id = 1 if not data else max([int(exp['experience_id']) for exp in data]) + 1

    new_experience = {
        "experience_id": str(experience_id),
        "steps": []  # Empty steps array; can be filled as needed
    }
    data.append(new_experience)
    write_json_file(file_path, data)



def add_step(file_path: str, step: Step) -> None:
    data = read_json_file(file_path)

    # Find the latest experience
    if not data:
        raise ValueError("No experience found. Please create a new experience first.")

    # Append the new step to the last experience in the list
    data[-1]['steps'].append(step)

    write_json_file(file_path, data)


def add_info(file_path: str, info: InfoDict) -> None:
    """Adds new info to the existing JSON data in the specified file."""
    data = read_json_file(file_path)

    # Check if data is empty and raise an error if so
    if not data:
        # raise ValueError("No existing data found. Please create a new entry first.")
        data = []

    # Append the new information
    data.append(info)

    # Write the updated data back to the file
    write_json_file(file_path, data)




def create_new_query(query:str, response:str) -> None:
    file_path = "memory/info.json"
    os.makedirs("memory", exist_ok=True)
    if not os.path.exists(file_path):
        # create the file if it does not exist
        with open(file_path, 'w') as file:
            json.dump([], file)

    data = read_json_file(file_path)

    # new query_id should be the maximum query_id + 1
    query_id = 1 if not data else max([query['query_id'] for query in data]) + 1

    # current time
    timestamp = datetime.now().isoformat()

    new_query = {"query_id": query_id, "query": query, "response": response, "timestamp": timestamp}
    data.append(new_query)
    write_json_file(file_path, data)

