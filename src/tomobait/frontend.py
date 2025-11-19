import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import requests

from .config import get_config, save_config
from .storage import Conversation, get_storage

# Load configuration
config = get_config()
BACKEND_URL = f"http://{config.server.backend_host}:{config.server.backend_port}/chat"
DOCS_DIR = os.path.abspath(config.documentation.sphinx_build_html_path)

# Storage
storage = get_storage()

# Global state for current conversation
current_conversation_id = None


def format_response(text):
    """
    This function takes the raw text response from the agent and formats it for
    display in the Gradio interface. It converts image paths to local URLs
    that Gradio can serve.
    """
    # Find all image paths (markdown or raw)
    image_paths = re.findall(
        r"!\[.*?\]\((.*?)\)|([\w\-/.]+\.(?:png|jpg|jpeg|gif|svg))", text
    )

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
            clean_text = re.sub(r"!\[.*?\]\(\)", "", parts[0]).strip()
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
    Uses the modern 'messages' format with role and content.
    """
    global current_conversation_id

    try:
        response = requests.post(BACKEND_URL, json={"query": message})
        response.raise_for_status()
        agent_response = response.json().get("response", "No response from agent.")

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Format agent response and add to history
        # For now, we'll just add the text response
        # Images will be embedded in the text if present
        history.append({"role": "assistant", "content": agent_response})

        return history

    except requests.exceptions.RequestException as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error connecting to backend: {e}"})
        return history


def new_conversation(history):
    """
    Save current conversation and start a new one.
    """
    global current_conversation_id

    # Auto-save current conversation if it has messages
    if history and len(history) > 0:
        # Convert messages format to storage format
        conv = Conversation()
        for msg in history:
            conv.add_message(msg["role"], msg["content"])
        conv.title = conv.generate_title()
        storage.save(conv)

    # Reset current conversation ID
    current_conversation_id = None

    # Return empty history and updated conversation list
    return [], get_conversation_list()


def get_conversation_list():
    """
    Get list of conversations for the history tab.
    Returns a formatted string for display.
    """
    conversations = storage.list_all()

    if not conversations:
        return "No saved conversations"

    lines = []
    for conv in conversations:
        lines.append(f"**{conv['title']}**")
        lines.append(f"ID: `{conv['id']}`")
        lines.append(f"Messages: {conv['message_count']}")
        lines.append(f"Updated: {conv['updated_at'][:19]}")
        lines.append(f"Preview: {conv['preview']}")
        lines.append("---")

    return "\n".join(lines)


def load_conversation_by_id(conv_id: str):
    """
    Load a conversation by ID.
    """
    global current_conversation_id

    conv = storage.load(conv_id.strip())
    if conv:
        current_conversation_id = conv.id
        # Convert to messages format
        history = [{"role": msg.role, "content": msg.content} for msg in conv.messages]
        return history, f"Loaded: {conv.title}"
    else:
        return [], f"Conversation not found: {conv_id}"


def delete_conversation_by_id(conv_id: str):
    """
    Delete a conversation by ID.
    """
    if storage.delete(conv_id.strip()):
        return get_conversation_list(), f"Deleted conversation: {conv_id}"
    else:
        return get_conversation_list(), f"Conversation not found: {conv_id}"


def load_config_values():
    """
    Load current config values for the config tab.
    """
    import yaml
    config = get_config()

    # Determine provider from api_type
    api_type_to_provider = {
        "google": "gemini",
        "openai": "openai",
        "azure": "azure",
        "anthropic": "anthropic",
        "anl_argo": "anl_argo"
    }
    provider = api_type_to_provider.get(config.llm.api_type, "gemini")

    # Format resources as YAML for display
    config_dict = config.model_dump()
    resources_yaml = ""
    if "resources" in config_dict:
        resources_yaml = yaml.dump(
            {"resources": config_dict["resources"]},
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )

    return (
        "\n".join(config.documentation.git_repos),
        "\n".join(config.documentation.local_folders),
        config.documentation.docs_output_dir,
        config.documentation.sphinx_build_html_path,
        config.retriever.db_path,
        config.retriever.embedding_model,
        config.retriever.k,
        config.retriever.search_type,
        config.retriever.score_threshold or 0.0,
        provider,  # Add provider
        config.llm.api_key_env,  # Add api_key_env
        config.llm.api_type,
        config.llm.model,
        config.llm.system_message,
        config.llm.anl_api_url or "",  # ANL Argo fields
        config.llm.anl_user or "",
        config.llm.anl_model or "",
        config.text_processing.chunk_size,
        config.text_processing.chunk_overlap,
        resources_yaml,  # Resources display
    )


def save_config_values(
    git_repos_str,
    local_folders_str,
    docs_output_dir,
    sphinx_build_html_path,
    db_path,
    embedding_model,
    k,
    search_type,
    score_threshold,
    provider,  # Add provider (not saved, just for UI)
    api_key_env,  # Add api_key_env
    api_type,
    model,
    system_message,
    anl_api_url,  # ANL Argo fields
    anl_user,
    anl_model,
    chunk_size,
    chunk_overlap,
):
    """
    Save config values from the config tab.
    """
    config = get_config()

    # Parse multi-line strings
    git_repos = [line.strip() for line in git_repos_str.split("\n") if line.strip()]
    local_folders = [
        line.strip() for line in local_folders_str.split("\n") if line.strip()
    ]

    # Update config
    config.documentation.git_repos = git_repos
    config.documentation.local_folders = local_folders
    config.documentation.docs_output_dir = docs_output_dir
    config.documentation.sphinx_build_html_path = sphinx_build_html_path

    config.retriever.db_path = db_path
    config.retriever.embedding_model = embedding_model
    config.retriever.k = k
    config.retriever.search_type = search_type
    config.retriever.score_threshold = score_threshold if score_threshold > 0 else None

    config.llm.api_key_env = api_key_env
    config.llm.model = model
    config.llm.api_type = api_type
    config.llm.system_message = system_message

    # Save ANL Argo fields
    config.llm.anl_api_url = anl_api_url if anl_api_url else None
    config.llm.anl_user = anl_user if anl_user else None
    config.llm.anl_model = anl_model if anl_model else None

    config.text_processing.chunk_size = chunk_size
    config.text_processing.chunk_overlap = chunk_overlap

    # Save to file
    save_config(config)

    return "Configuration saved successfully! Config will hot-reload automatically."


def check_vectordb_status():
    """
    Check the status of the vector database.
    Returns a status message indicating the number of documents.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from .config import get_config

        config = get_config()
        embeddings = HuggingFaceEmbeddings(model_name=config.retriever.embedding_model)
        vectorstore = Chroma(
            persist_directory=config.retriever.db_path,
            embedding_function=embeddings
        )
        count = vectorstore._collection.count()

        if count == 0:
            return "⚠️ Empty - No documents indexed. Please run ingestion."
        else:
            return f"✅ Ready - {count} documents indexed"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_data_ingestion():
    """
    Run the data ingestion process.
    Yields progress updates as the process runs.
    """
    import subprocess
    import os

    yield "🔄 Starting data ingestion process...\n"

    try:
        # Run the ingestion command
        cmd = ["/home/raf/.pixi/bin/pixi", "run", "ingest"]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd()
        )

        output = ""
        for line in process.stdout:
            output += line
            yield output

        process.wait()

        if process.returncode == 0:
            # Check final status
            status = check_vectordb_status()
            yield output + f"\n\n✅ Ingestion completed successfully!\n{status}"
        else:
            yield output + f"\n\n❌ Ingestion failed with exit code {process.returncode}"

    except Exception as e:
        yield f"❌ Error running ingestion: {str(e)}"


def update_llm_fields_from_provider(provider):
    """
    Update LLM-related fields based on selected provider.
    Returns: (model_choices, api_type, api_key_env, anl_settings_visibility)
    """
    provider_config = {
        "gemini": {
            "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            "api_type": "google",
            "api_key_env": "GEMINI_API_KEY"
        },
        "openai": {
            "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
            "api_type": "openai",
            "api_key_env": "OPENAI_API_KEY"
        },
        "azure": {
            "models": ["gpt-4", "gpt-35-turbo"],
            "api_type": "azure",
            "api_key_env": "AZURE_OPENAI_API_KEY"
        },
        "anthropic": {
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "api_type": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "anl_argo": {
            "models": ["llama-2-70b", "mixtral-8x7b"],
            "api_type": "anl_argo",
            "api_key_env": ""  # No API key needed for ANL Argo
        }
    }

    config = provider_config.get(provider, provider_config["gemini"])

    # Show ANL settings only for ANL Argo provider
    anl_visible = (provider == "anl_argo")

    return (
        gr.Dropdown(choices=config["models"], value=config["models"][0]),
        config["api_type"],
        config["api_key_env"],
        gr.Group(visible=anl_visible)  # Show/hide ANL settings
    )


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# TomoBait Chat")
    gr.Markdown("Ask questions about the 2-BM beamline documentation.")

    with gr.Tabs():
        # --- Tab 1: Chat Interface ---
        with gr.Tab("Chat"):
            with gr.Row():
                new_chat_btn = gr.Button("🆕 New Conversation", size="sm")

            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                height=500,
                type="messages",
            )

            with gr.Row():
                txt = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Enter your question and press enter",
                    container=False,
                )

            # Connect chat function
            txt.submit(chat_func, [txt, chatbot], [chatbot]).then(
                lambda: "", None, txt
            )  # Clear input

            # Connect new conversation button
            new_chat_btn.click(new_conversation, [chatbot], [chatbot, gr.State()])

        # --- Tab 2: Chat History ---
        with gr.Tab("History"):
            gr.Markdown("## Saved Conversations")

            history_display = gr.Markdown(get_conversation_list())

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                refresh_btn.click(
                    lambda: get_conversation_list(), None, history_display
                )

            gr.Markdown("### Load Conversation")
            with gr.Row():
                load_id_input = gr.Textbox(
                    label="Conversation ID",
                    placeholder="Enter conversation ID to load",
                    scale=3,
                )
                load_btn = gr.Button("📥 Load", size="sm", scale=1)

            load_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Delete Conversation")
            with gr.Row():
                delete_id_input = gr.Textbox(
                    label="Conversation ID",
                    placeholder="Enter conversation ID to delete",
                    scale=3,
                )
                delete_btn = gr.Button("🗑️ Delete", size="sm", scale=1)

            delete_status = gr.Textbox(label="Status", interactive=False)

            # Connect history functions
            load_btn.click(
                load_conversation_by_id, [load_id_input], [chatbot, load_status]
            )
            delete_btn.click(
                delete_conversation_by_id,
                [delete_id_input],
                [history_display, delete_status],
            )

        # --- Tab 3: Configuration ---
        with gr.Tab("Configuration"):
            gr.Markdown("## Application Configuration")
            gr.Markdown(
                "Edit settings and click Save. **Restart the backend** to apply changes."
            )

            with gr.Accordion("📚 Documentation Sources", open=True):
                git_repos = gr.Textbox(
                    label="Git Repositories (one per line)",
                    lines=3,
                    placeholder="https://github.com/user/repo.git",
                )
                local_folders = gr.Textbox(
                    label="Local Folders (one per line)",
                    lines=3,
                    placeholder="/path/to/docs",
                )
                docs_output = gr.Textbox(label="Documentation Output Directory")
                sphinx_html = gr.Textbox(label="Sphinx HTML Build Path")

            with gr.Accordion("🔍 Retriever Settings", open=True):
                db_path = gr.Textbox(label="ChromaDB Path")
                embedding_model = gr.Textbox(label="Embedding Model")
                k_value = gr.Slider(
                    label="Number of Documents (k)", minimum=1, maximum=20, step=1
                )
                search_type = gr.Dropdown(
                    label="Search Type",
                    choices=["similarity", "mmr", "similarity_score_threshold"],
                )
                score_threshold = gr.Slider(
                    label="Score Threshold (0 = disabled)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                )

            with gr.Accordion("🤖 LLM Settings", open=True):
                provider = gr.Dropdown(
                    label="LLM Provider",
                    choices=["gemini", "openai", "azure", "anthropic", "anl_argo"],
                    value="gemini",
                )
                api_key_env = gr.Dropdown(
                    label="API Key Environment Variable",
                    choices=["GEMINI_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
                    value="GEMINI_API_KEY",
                    allow_custom_value=True,
                )
                api_type = gr.Dropdown(
                    label="API Type (for autogen)",
                    choices=["google", "openai", "azure", "anthropic", "anl_argo"],
                    value="google",
                    allow_custom_value=True,
                )
                llm_model = gr.Dropdown(
                    label="Model Name",
                    choices=[
                        "gemini-2.5-flash",
                        "gemini-2.0-flash",
                        "gemini-1.5-pro",
                        "gpt-4",
                        "gpt-4-turbo",
                        "gpt-4o",
                        "gpt-3.5-turbo",
                        "claude-3-opus",
                        "claude-3-sonnet",
                        "claude-3-haiku",
                        "llama-2-70b",
                        "mixtral-8x7b",
                    ],
                    value="gemini-2.5-flash",
                    allow_custom_value=True,
                )
                system_msg = gr.Textbox(label="System Message", lines=5)

                # ANL Argo specific settings (shown/hidden based on provider)
                with gr.Group(visible=False) as anl_settings:
                    gr.Markdown("### ANL Argo Configuration")
                    anl_api_url = gr.Textbox(
                        label="ANL API URL",
                        placeholder="https://your-anl-argo-endpoint/api/llm",
                        value="",
                    )
                    anl_user = gr.Textbox(
                        label="ANL Username",
                        placeholder="your_anl_username",
                        value="",
                    )
                    anl_model = gr.Textbox(
                        label="ANL Model Name",
                        placeholder="llama-2-70b",
                        value="",
                    )

            with gr.Accordion("✂️ Text Processing", open=False):
                chunk_size = gr.Slider(
                    label="Chunk Size", minimum=100, maximum=5000, step=100
                )
                chunk_overlap = gr.Slider(
                    label="Chunk Overlap", minimum=0, maximum=1000, step=50
                )

            with gr.Accordion("📚 Resources (APS Tomography)", open=False):
                gr.Markdown(
                    "This section contains links to beamlines, software packages, "
                    "and community resources. Edit config.yaml directly to modify."
                )
                resources_display = gr.Code(
                    label="Resources Configuration",
                    language="yaml",
                    interactive=False,
                    lines=20,
                )

            with gr.Accordion("🔄 Data Ingestion", open=True):
                gr.Markdown(
                    "Ingest documentation into the vector database. "
                    "Run this after changing documentation sources or to refresh the index."
                )
                with gr.Row():
                    ingest_status_display = gr.Textbox(
                        label="Vector Database Status",
                        value="Checking...",
                        interactive=False,
                    )
                ingest_btn = gr.Button("🚀 Run Data Ingestion", variant="secondary")
                ingest_output = gr.Textbox(
                    label="Ingestion Log",
                    interactive=False,
                    lines=10,
                    placeholder="Ingestion output will appear here...",
                )

            save_cfg_btn = gr.Button("💾 Save Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False)

            # Connect provider change event to update fields
            provider.change(
                update_llm_fields_from_provider,
                inputs=[provider],
                outputs=[llm_model, api_type, api_key_env, anl_settings],
            )

            # Load current config values on startup
            demo.load(
                load_config_values,
                None,
                [
                    git_repos,
                    local_folders,
                    docs_output,
                    sphinx_html,
                    db_path,
                    embedding_model,
                    k_value,
                    search_type,
                    score_threshold,
                    provider,
                    api_key_env,
                    api_type,
                    llm_model,
                    system_msg,
                    anl_api_url,
                    anl_user,
                    anl_model,
                    chunk_size,
                    chunk_overlap,
                    resources_display,
                ],
            )

            # Connect save config function
            save_cfg_btn.click(
                save_config_values,
                [
                    git_repos,
                    local_folders,
                    docs_output,
                    sphinx_html,
                    db_path,
                    embedding_model,
                    k_value,
                    search_type,
                    score_threshold,
                    provider,
                    api_key_env,
                    api_type,
                    llm_model,
                    system_msg,
                    anl_api_url,
                    anl_user,
                    anl_model,
                    chunk_size,
                    chunk_overlap,
                ],
                config_status,
            )

            # Connect ingestion button
            ingest_btn.click(
                run_data_ingestion,
                None,
                ingest_output
            ).then(
                check_vectordb_status,
                None,
                ingest_status_display
            )

            # Load initial ingestion status
            demo.load(
                check_vectordb_status,
                None,
                ingest_status_display
            )

        # --- Tab 4: Setup (AI Config Generator) ---
        with gr.Tab("Setup"):
            gr.Markdown("## AI-Powered Configuration Generator")
            gr.Markdown(
                "Describe your configuration needs in natural language, and AI will generate a config for you."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Describe Your Needs")
                    user_prompt = gr.Textbox(
                        label="Configuration Prompt",
                        lines=8,
                        placeholder=(
                            "Example: I want to index local documentation in /data/tomo "
                            "with high accuracy retrieval. I need 5 documents per query and "
                            "want to use smaller chunks for better precision."
                        ),
                    )

                    gr.Markdown("**Example prompts:**")
                    gr.Markdown(
                        "- *Index GitHub repo https://github.com/user/docs with default settings*\n"
                        "- *High-speed setup with 2 retrieval docs and large chunks*\n"
                        "- *Index multiple local folders: /data/tomo1, /data/tomo2, /data/tomo3*"
                    )

                    generate_btn = gr.Button(
                        "🤖 Generate Configuration", variant="primary", size="lg"
                    )
                    gen_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Generated Configuration Preview")
                    generated_yaml = gr.Code(
                        label="YAML Preview",
                        language="yaml",
                        lines=20,
                        interactive=True,
                    )

                    with gr.Row():
                        apply_btn = gr.Button("✅ Apply Configuration", variant="primary")
                        cancel_btn = gr.Button("❌ Cancel")

                    apply_status = gr.Textbox(label="Apply Status", interactive=False)

            # Functions for Setup tab
            def generate_config_ui(prompt):
                """Generate config from user prompt."""
                if not prompt or not prompt.strip():
                    return "", "Please provide a configuration prompt"

                try:
                    # Call backend to generate config
                    response = requests.post(
                        f"http://{config.server.backend_host}:{config.server.backend_port}/generate-config",
                        json={"prompt": prompt},
                    )
                    response.raise_for_status()

                    result = response.json()
                    yaml_config = result["yaml_config"]

                    return yaml_config, "✅ Configuration generated successfully! Review and click Apply."

                except requests.exceptions.RequestException as e:
                    return "", f"❌ Error: {str(e)}"
                except Exception as e:
                    return "", f"❌ Unexpected error: {str(e)}"

            def apply_generated_config(yaml_str):
                """Apply the generated configuration."""
                if not yaml_str or not yaml_str.strip():
                    return "No configuration to apply"

                try:
                    # Parse YAML
                    import yaml

                    config_dict = yaml.safe_load(yaml_str)

                    # Send to backend to apply
                    response = requests.post(
                        f"http://{config.server.backend_host}:{config.server.backend_port}/apply-config",
                        json=config_dict,
                    )
                    response.raise_for_status()

                    result = response.json()
                    backup_path = result.get("backup_path", "")

                    return (
                        f"✅ Configuration applied successfully!\n"
                        f"Backup saved to: {backup_path}\n"
                        f"Backend will reload automatically."
                    )

                except yaml.YAMLError as e:
                    return f"❌ Invalid YAML: {str(e)}"
                except requests.exceptions.RequestException as e:
                    return f"❌ Error applying config: {str(e)}"
                except Exception as e:
                    return f"❌ Unexpected error: {str(e)}"

            def clear_generated():
                """Clear the generated config."""
                return "", "Cancelled"

            # Connect Setup tab functions
            generate_btn.click(
                generate_config_ui, [user_prompt], [generated_yaml, gen_status]
            )
            apply_btn.click(apply_generated_config, [generated_yaml], apply_status)
            cancel_btn.click(clear_generated, None, [generated_yaml, apply_status])

if __name__ == "__main__":
    demo.launch(
        server_name=config.server.frontend_host,
        server_port=config.server.frontend_port,
        allowed_paths=[DOCS_DIR],  # This is crucial for serving images
    )
