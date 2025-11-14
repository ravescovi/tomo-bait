# import os
# import autogen  # This is the correct v2 import
# from autogen import LLMConfig
# from typing import Annotated
# from retriever import get_documentation_retriever
# from dotenv import load_dotenv

# load_dotenv()

# # --- 1. Interchangeable LLM Config (NEW FORMAT) ---
# # We check for the standard GEMINI_API_KEY
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     print("❌ ERROR: GEMINI_API_KEY environment variable not set.")
#     exit()

# # Use the new config OBJECT, not an old config_list
# llm_config = LLMConfig(
#     config_list={
#         "api_type": "google",
#         "model": "gemini-2.0-flash-lite",
#         "api_key": os.getenv("GEMINI_API_KEY"),
#     }
# )

# # --- 2. Load Retriever (from Phase 1) ---
# # This part was perfect.
# retriever = get_documentation_retriever()


# # --- 3. Define Agents ---
# # This was mostly correct, just needs the new llm_config
# technician_agent = autogen.AssistantAgent(
#     "doc_expert",
#     llm_config=llm_config,  # <-- Pass the config OBJECT here
#     system_message=(
#         "You are an expert on this project's documentation. "
#         "A user will ask a question. Your 'query_documentation' tool "
#         "will provide you with the *only* relevant context. "
#         "**You must answer the user's question based *only* on that context.** "
#         "If the context is not sufficient, say so. Do not make up answers."
#     )
# )

# worker_agent = autogen.UserProxyAgent(
#     "tool_worker",
#     llm_config=False,
#     code_execution_config=False,
# )


# # --- 4. Define & Register the "Tool" (NEW FORMAT) ---
# # We use decorators to register the function to the agents.
# # This replaces the old global `autogen.register_function`.

# @technician_agent.register_for_llm(
#     name="query_documentation",
#     description="Search the project documentation for a given query."
# )
# @worker_agent.register_for_execution(name="query_documentation")
# def query_documentation(
#     query: Annotated[str, "The search query for the documentation"]
# ) -> str:
#     """
#     A tool that takes a user's query, retrieves relevant
#     document chunks, and returns them as a single string.
#     """
#     print(f"\n--- TOOL: Querying for '{query}' ---")
    
#     results = retriever.invoke(query)
    
#     context_str = "\n\n---\n\n".join(
#         [doc.page_content for doc in results]
#     )
    
#     print(f"--- TOOL: Found {len(results)} chunks. ---")
#     return context_str

# # --- 5. Start the Chat ---
# # This part was perfect.
# print("Starting agent chat...")
# USER_QUESTION = "How do I align my beam?"

# chat_result = worker_agent.initiate_chat(
#     recipient=technician_agent,
#     message=(
#         f"Please answer this question: '{USER_QUESTION}'. "
#         "You *must* use the 'query_documentation' tool to "
#         "find the relevant context first."
#     ),
#     max_turns=2
# )

import os
import autogen
from autogen import LLMConfig
from typing import Annotated
from retriever import get_documentation_retriever
from dotenv import load_dotenv

# --- FIX 1: Remove the Google-specific type imports ---
# We will use a plain dict, so we don't need these.
# from google.generativeai.types import Tool
# from google.ai.generativelanguage import FunctionDeclaration, Schema, Type


load_dotenv()

# --- 1. Interchangeable LLM Config ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY environment variable not set.")
    exit()

# --- FIX 2: Define the tool as a standard Python dict (OpenAI format) ---
# This is what the autogen wrapper expects to receive.
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


# Use the generic LLMConfig
llm_config = LLMConfig(
    config_list=[
        {
            "api_type": "google",
            "model": "gemini-2.0-flash-lite",
            "api_key": api_key,
            # --- FIX 3: Pass the list containing the tool DICT ---
            "tools": [query_documentation_tool_dict],
        }
    ]
)

# --- 2. Load Retriever (from Phase 1) ---
retriever = get_documentation_retriever()


# --- 3. Define Agents ---
technician_agent = autogen.AssistantAgent(
    "doc_expert",
    llm_config=llm_config,  # <-- This config now contains the tool dict
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
    code_execution_config=False,
)


# --- 4. Define & Register the "Tool" ---
# This part is correct:
# - No @register_for_llm (handled by llm_config)
# - YES @register_for_execution (so the worker can run it)

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

# --- 5. Start the Chat ---
# (This section is perfect, no changes needed)
print("Starting agent chat...")
USER_QUESTION = "How do I align my beam?"

chat_result = worker_agent.initiate_chat(
    recipient=technician_agent,
    message=(
        f"Please answer this question: '{USER_QUESTION}'. "
        "You *must* use the 'query_documentation' tool to "
        "find the relevant context first."
    ),
    max_turns=2
)

# Print the final answer
if chat_result and chat_result.chat_history:
    print("\n--- FINAL ANSWER ---")
    print(chat_result.chat_history[-1]["content"])