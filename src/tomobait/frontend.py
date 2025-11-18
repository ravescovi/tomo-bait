import gradio as gr
import requests
import re
import os

BACKEND_URL = "http://127.0.0.1:8001/chat"
DOCS_DIR = os.path.abspath("tomo_documentation/2bm-docs/docs/_build/html")

def format_response(text):
    """
    This function takes the raw text response from the agent and formats it for
    display in the Gradio interface. It converts image paths to local URLs
    that Gradio can serve.
    """
    # Find all image paths (markdown or raw)
    image_paths = re.findall(r'!\[.*?\]\((.*?)\)|([\w\-/.]+\.(?:png|jpg|jpeg|gif|svg))', text)
    
    # Flatten the list of tuples from findall
    flat_paths = [item for sublist in image_paths for item in sublist if item]
    
    # Create a list of tuples (original_text, image_path)
    # to be used in the Gradio chatbot component
    output_components = []
    
    # Start with the full text
    remaining_text = text
    
    for path in flat_paths:
        # The path from regex might be inside markdown, e.g., `_images/my-image.png`
        # We split the text by the image path to insert the image
        parts = remaining_text.split(path, 1)
        
        # Add the text before the image
        if parts[0].strip():
            # also remove the markdown remnant `![]()`
            clean_text = re.sub(r'!\[.*?\]\(\)', '', parts[0]).strip()
            if clean_text:
                output_components.append((clean_text, None))

        # Add the image
        # Gradio needs an absolute path to serve the file
        full_image_path = os.path.join(DOCS_DIR, path)
        if os.path.exists(full_image_path):
            output_components.append((None, full_image_path))
        else:
            # If the image path is broken, just append the text
             output_components.append((f"(Image not found: {path})", None))

        # The rest of the text
        remaining_text = parts[1] if len(parts) > 1 else ""

    # Add any remaining text after the last image
    if remaining_text.strip():
        output_components.append((remaining_text.strip(), None))

    # If no images were found, just return the original text
    if not output_components:
        return [(text, None)]
        
    return output_components

def chat_func(message, history):
    """
    This is the function that Gradio calls when the user sends a message.
    """
    try:
        response = requests.post(BACKEND_URL, json={"query": message})
        response.raise_for_status()
        agent_response = response.json().get("response", "No response from agent.")
        
        # We give the user message back to Gradio to display it
        # Then we format the agent's response to handle images
        formatted_agent_response = format_response(agent_response)
        
        # The history in Gradio is a list of lists,
        # e.g., [["user message", "agent response"], ...]
        # We will append the new turn to the history
        history.append((message, None)) # User message
        for text_part, image_part in formatted_agent_response:
            history.append((text_part, image_part))

        return history

    except requests.exceptions.RequestException as e:
        history.append((message, f"Error connecting to backend: {e}"))
        return history

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# TomoBait Chat")
    gr.Markdown("Ask questions about the 2-BM beamline documentation.")

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        height=500,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter your question and press enter",
            container=False,
        )
    
    txt.submit(chat_func, [txt, chatbot], [chatbot])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        allowed_paths=[DOCS_DIR] # This is crucial for serving images
    )