import os
from typing import Annotated

import autogen
from autogen import LLMConfig
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .retriever import get_documentation_retriever

load_dotenv()

# --- FastAPI App ---
api = FastAPI()

class ChatQuery(BaseModel):
    query: str

@api.post("/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to receive a query and return the agent's response.
    """
    answer = run_agent_chat(chat_query.query)
    return {"response": answer}

# --- 1. Interchangeable LLM Config ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY environment variable not set.")
    exit()

query_documentation_tool_dict = {
    "type": "function",
    "function": {
        "name": "query_documentation",
        "description": "Search the project documentation for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for the documentation"
                }
            },
            "required": ["query"]
        }
    }
}

llm_config = LLMConfig(
    config_list=[
        {
            "api_type": "google",
            "model": "gemini-2.5-flash",
            "api_key": api_key,
            "tools": [query_documentation_tool_dict],
        }
    ]
)

# --- 2. Load Retriever (from Phase 1) ---
retriever = get_documentation_retriever()


# --- 3. Define Agents ---
technician_agent = autogen.AssistantAgent(
    "doc_expert",
    llm_config=llm_config,
    system_message=(
        "You are an expert on this project's documentation. "
        "A user will ask a question. Your 'query_documentation' tool "
        "will provide you with the *only* relevant context. "
        "**You must answer the user's question based *only* on that context.** "
        "If the context is not sufficient, say so. Do not make up answers."
    )
)

worker_agent = autogen.UserProxyAgent(
    "tool_worker",
    llm_config=False,
    human_input_mode="NEVER",
    # Terminate the conversation when the other agent sends a message
    # that doesn't contain a tool call.
    is_termination_msg=lambda msg: not msg.get("tool_calls"),
    code_execution_config=False,
)

@worker_agent.register_for_execution(name="query_documentation")
def query_documentation(
    query: Annotated[str, "The search query for the documentation"]
) -> str:
    """
    A tool that takes a user's query, retrieves relevant
    document chunks, and returns them as a single string.
    """
    print(f"\n--- TOOL: Querying for '{query}' ---")
    
    results = retriever.invoke(query)
    
    context_str = "\n\n---\n\n".join(
        [doc.page_content for doc in results]
    )
    
    print(f"--- TOOL: Found {len(results)} chunks. ---")
    return context_str

def run_agent_chat(user_question: str) -> str:
    """
    Initializes and runs a chat between agents to answer a user's question.
    """
    print("Starting agent chat...")
    chat_result = worker_agent.initiate_chat(
        recipient=technician_agent,
        message=(
            f"Please answer this question: '{user_question}'. "
            "You *must* use the 'query_documentation' tool to "
            "find the relevant context first."
        ),
    )

    # The summary is the last message that was sent in the chat.
    # In our case, this is the final answer from the technician agent.
    final_answer = chat_result.summary
    if final_answer:
        print("\n--- FINAL ANSWER ---")
        print(final_answer)
        return final_answer
    return "Sorry, I couldn't find an answer."


if __name__ == '__main__':
    USER_QUESTION = "What epics devices are connected to the beamline"
    run_agent_chat(USER_QUESTION)
