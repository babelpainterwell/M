from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
import json


def retrieve_info_memory(request, relevance_threshold=0.6):
    loader = JSONLoader(file_path="memory/info.json", jq_schema=".[]", text_content=False)
    documents = loader.load()

    if not documents:
        raise ValueError("No documents found")

    embedding = OpenAIEmbeddings()

    db = Chroma.from_documents(documents, embedding)
    docs = db.similarity_search_with_relevance_scores(request, k=1)

    # Check if there are any results
    if not docs or not docs[0]:
        raise ValueError("No matching documents found")

    if docs[0][1] < relevance_threshold:
        raise ValueError("No sufficiently relevant documents found")
    
    return docs[0]


def retrieve_experience_memory(bbox_observation, visual_summary, k=5):
    """
    Inputs: 
    bbox_observation: str
    visual_summary: str

    Outputs:
    List of Langchain Document
    """
    loader = JSONLoader(file_path="memory/experiences.json", jq_schema=".[].steps[]", text_content=False)
    documents = loader.load()

    context = {
        "bbox_observation": bbox_observation,
        "visual_summary": visual_summary,
    }

    context = json.dumps(context)

    if not documents:
        raise ValueError("No documents found")

    embedding = OpenAIEmbeddings()

    db = Chroma.from_documents(documents, embedding)
    docs = db.similarity_search(context, k=k)


    if not docs or not docs[0]:
        raise ValueError("No matching documents found")
    

    # return a list of Langchain Document, in which each Document contains one step
    return docs