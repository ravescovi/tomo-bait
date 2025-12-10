import os
import re
from pathlib import Path

import gradio as gr
import requests

from .config import get_config, save_config
from .storage import Conversation, get_storage

# Load configuration
config = get_config()
BACKEND_BASE = f"http://{config.server.backend_host}:{config.server.backend_port}"
BACKEND_URL = f"{BACKEND_BASE}/chat"
HEALTH_URL = f"{BACKEND_BASE}/health"
sphinx_path = config.get_sphinx_build_html_path()
DOCS_DIR = os.path.abspath(str(sphinx_path)) if sphinx_path else None

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

        # Handle specific error codes
        if response.status_code == 503:
            error_data = response.json().get("detail", {})
            is_llm_error = (
                isinstance(error_data, dict)
                and error_data.get("error") == "llm_not_configured"
            )
            if is_llm_error:
                # LLM not configured - give helpful message
                hint = error_data.get("hint", "Check your API key configuration")
                provider = error_data.get("provider", "unknown")
                model = error_data.get("model", "unknown")
                error_msg = (
                    f"**LLM Not Configured**\n\n"
                    f"The {provider}/{model} provider is not available.\n\n"
                    f"**To fix this:**\n"
                    f"- {hint}\n"
                    f"- Or switch to a different provider in the Configuration tab"
                )
            else:
                error_msg = "The AI service is overloaded. Please try again shortly."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history

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
        history.append({
            "role": "assistant",
            "content": f"Error connecting to backend: {e}"
        })
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


def generate_conversation_list_html():
    """
    Generate HTML for the conversation list with inline Download/Delete buttons.
    """
    try:
        conversations = storage.list_all()
    except Exception as e:
        return f"<p style='color: #d32f2f; padding: 20px;'>Error loading conversations: {str(e)}</p>"

    if not conversations:
        return "<p style='color: #666; padding: 20px;'>No saved conversations</p>"

    html = """
    <style>
        .conv-list {
            width: 100%;
            border-collapse: collapse;
            font-family: system-ui, -apple-system, sans-serif;
        }
        .conv-row {
            border-bottom: 1px solid #e0e0e0;
            transition: background-color 0.2s;
        }
        .conv-row:hover {
            background-color: #f5f5f5;
        }
        .conv-title {
            cursor: pointer;
            color: #1976d2;
            font-weight: 500;
            padding: 12px 8px;
        }
        .conv-title:hover {
            color: #1565c0;
            text-decoration: underline;
        }
        .conv-meta {
            color: #666;
            font-size: 0.875rem;
            padding: 4px 8px;
        }
        .conv-actions {
            padding: 8px;
            text-align: right;
            white-space: nowrap;
        }
        .conv-btn {
            padding: 6px 12px;
            margin-left: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        .conv-btn:hover {
            background: #f0f0f0;
            border-color: #999;
        }
        .conv-btn.download {
            color: #1976d2;
        }
        .conv-btn.delete {
            color: #d32f2f;
        }
        .conv-btn.delete:hover {
            background: #ffebee;
        }
    </style>
    <table class="conv-list">
    """

    for conv in conversations:
        conv_id = conv['id']
        title = conv['title']
        message_count = conv['message_count']
        updated = conv['updated_at'][:19].replace('T', ' ')
        preview = conv['preview'][:80] + ('...' if len(conv['preview']) > 80 else '')

        html += f"""
        <tr class="conv-row">
            <td>
                <div class="conv-title" onclick="loadConversation('{conv_id}')">{title}</div>
                <div class="conv-meta">
                    {message_count} messages • Updated: {updated}
                    <br/>
                    <span style="color: #999;">{preview}</span>
                </div>
            </td>
            <td class="conv-actions">
                <button class="conv-btn download" onclick="downloadConversation('{conv_id}')">📥 Download</button>
                <button class="conv-btn delete" onclick="deleteConversation('{conv_id}')">🗑️ Delete</button>
            </td>
        </tr>
        """

    html += """
    </table>
    <script>
        function loadConversation(convId) {
            // Find the load button and input field
            const loadInput = document.querySelector('input[placeholder*="will be loaded"]');
            const loadBtn = document.querySelector('button:has-text("Load Selected")') ||
                           Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Load'));

            if (loadInput) {
                loadInput.value = convId;
                loadInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }

        function downloadConversation(convId) {
            const downloadInput = document.querySelector('input[placeholder*="download"]');
            const downloadBtn = Array.from(document.querySelectorAll('button')).find(b =>
                b.textContent.includes('Download') && b.textContent.includes('JSON'));

            if (downloadInput && downloadBtn) {
                downloadInput.value = convId;
                downloadInput.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => downloadBtn.click(), 100);
            }
        }

        function deleteConversation(convId) {
            if (confirm('Are you sure you want to delete this conversation?')) {
                const deleteInput = document.querySelector('input[placeholder*="delete"]');
                const deleteBtn = Array.from(document.querySelectorAll('button')).find(b =>
                    b.textContent.includes('Delete') && b.textContent.includes('Selected'));

                if (deleteInput && deleteBtn) {
                    deleteInput.value = convId;
                    deleteInput.dispatchEvent(new Event('input', { bubbles: true }));
                    setTimeout(() => deleteBtn.click(), 100);
                }
            }
        }
    </script>
    """

    return html


def show_conversation_details(conv_id: str):
    """
    Show full conversation details in a formatted display.
    """
    if not conv_id or not conv_id.strip():
        return "Select a conversation to view details."

    conv = storage.load(conv_id.strip())
    if not conv:
        return f"Conversation not found: {conv_id}"

    # Format conversation as markdown
    lines = [
        f"# {conv.title}",
        f"**ID:** `{conv.id}`",
        f"**Created:** {conv.created_at[:19].replace('T', ' ')}",
        f"**Updated:** {conv.updated_at[:19].replace('T', ' ')}",
        f"**Messages:** {len(conv.messages)}",
        "",
        "---",
        ""
    ]

    for i, msg in enumerate(conv.messages, 1):
        role_emoji = "👤" if msg.role == "user" else "🤖"
        lines.append(f"### {role_emoji} Message {i} ({msg.role})")
        lines.append(f"*{msg.timestamp[:19].replace('T', ' ')}*")
        lines.append("")
        lines.append(msg.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def download_conversation_json(conv_id: str):
    """
    Export conversation to JSON file.
    Returns the file path for download.
    """
    import json
    import tempfile
    from pathlib import Path

    if not conv_id or not conv_id.strip():
        return None, "No conversation ID provided"

    conv = storage.load(conv_id.strip())
    if not conv:
        return None, f"Conversation not found: {conv_id}"

    # Create temporary file
    temp_dir = Path(tempfile.gettempdir())
    filename = f"conversation_{conv.id}_{conv.title[:30].replace(' ', '_')}.json"
    filepath = temp_dir / filename

    # Convert to dict and save
    conv_dict = conv.model_dump()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conv_dict, f, indent=2, ensure_ascii=False)

    return str(filepath), f"Downloaded: {filename}"


def delete_conversation_and_refresh(conv_id: str):
    """
    Delete a conversation and refresh the list.
    """
    if not conv_id or not conv_id.strip():
        return generate_conversation_list_html(), "", "No conversation ID provided"

    conv = storage.load(conv_id.strip())
    if not conv:
        return generate_conversation_list_html(), "", f"Conversation not found: {conv_id}"

    title = conv.title
    if storage.delete(conv_id.strip()):
        return (
            generate_conversation_list_html(),
            "",
            f"✅ Deleted: {title}"
        )
    else:
        return (
            generate_conversation_list_html(),
            "",
            f"❌ Failed to delete: {conv_id}"
        )


def load_conversation_and_refresh(conv_id: str):
    """
    Load a conversation and return it for the chatbot.
    """
    global current_conversation_id

    if not conv_id or not conv_id.strip():
        return [], "", "No conversation ID provided"

    conv = storage.load(conv_id.strip())
    if conv:
        current_conversation_id = conv.id
        # Convert to messages format
        history = [{"role": msg.role, "content": msg.content} for msg in conv.messages]
        return history, "", f"✅ Loaded: {conv.title}"
    else:
        return [], "", f"❌ Conversation not found: {conv_id}"


def load_config_values():
    """
    Load current config values for Configuration tab (technical settings only).
    """
    config = get_config()

    # Determine provider from api_type and base_url
    api_type_to_provider = {
        "google": "gemini",
        "openai": "openai",
        "azure": "azure",
        "anthropic": "anthropic",
    }
    # Check if using Argo (OpenAI with Argo base_url)
    if config.llm.base_url and "argoapi" in config.llm.base_url:
        provider = "anl_argo"
    else:
        provider = api_type_to_provider.get(config.llm.api_type, "gemini")

    return (
        str(config.get_db_path()),
        config.retriever.embedding_model,
        config.retriever.k,
        config.retriever.search_type,
        config.retriever.score_threshold or 0.0,
        provider,
        config.llm.api_key_env,
        config.llm.api_type,
        config.llm.model,
        config.llm.system_message,
        config.llm.base_url or "",
        config.text_processing.chunk_size,
        config.text_processing.chunk_overlap,
    )


def load_sources_values():
    """
    Load documentation sources values for Sources tab.
    """
    import yaml
    config = get_config()

    # Format resources as YAML for display
    resources_yaml = ""
    if config.documentation.resources:
        resources_yaml = yaml.dump(
            {"resources": config.documentation.resources},
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )

    return (
        "\n".join(config.documentation.git_repos),
        "\n".join(config.documentation.local_folders),
        str(config.get_docs_output_dir()),
        str(config.get_sphinx_build_html_path()) if config.get_sphinx_build_html_path() else "",
        resources_yaml,
    )


def save_config_values(
    db_path,
    embedding_model,
    k,
    search_type,
    score_threshold,
    provider,  # Not saved, just for UI
    api_key_env,
    api_type,
    model,
    system_message,
    base_url,
    chunk_size,
    chunk_overlap,
):
    """
    Save technical config values from the Configuration tab.
    """
    config = get_config()

    # Update retriever settings
    config.retriever.db_path = db_path  # Note: This sets the override, computed path from get_db_path()
    config.retriever.embedding_model = embedding_model
    config.retriever.k = k
    config.retriever.search_type = search_type
    config.retriever.score_threshold = score_threshold if score_threshold > 0 else None

    # Update LLM settings
    config.llm.api_key_env = api_key_env
    config.llm.model = model
    config.llm.api_type = api_type
    config.llm.system_message = system_message
    config.llm.base_url = base_url if base_url else None

    # Update text processing settings
    config.text_processing.chunk_size = chunk_size
    config.text_processing.chunk_overlap = chunk_overlap

    # Save to file
    save_config(config)

    return "Configuration saved successfully! Config will hot-reload automatically."


def save_sources_values(
    git_repos_str,
    local_folders_str,
    docs_output_dir,
    sphinx_build_html_path,
):
    """
    Save documentation sources from the Sources tab.
    """
    config = get_config()

    # Parse multi-line strings
    git_repos = [line.strip() for line in git_repos_str.split("\n") if line.strip()]
    local_folders = [
        line.strip() for line in local_folders_str.split("\n") if line.strip()
    ]

    # Update documentation sources
    config.documentation.git_repos = git_repos
    config.documentation.local_folders = local_folders
    config.documentation.docs_output_dir = docs_output_dir
    config.documentation.sphinx_build_html_path = sphinx_build_html_path

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
        from .config import get_config
        from .embeddings import get_embeddings

        config = get_config()
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=str(config.get_db_path()),
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
        cmd = ["uv", "run", "ingest"]

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


def get_sources_summary():
    """
    Get a summary of all configured sources and their status.
    Returns formatted markdown string with statistics.
    """
    config = get_config()

    # Count sources
    git_repo_count = len(config.documentation.git_repos)
    local_folder_count = len(config.documentation.local_folders)

    # Count resources
    resource_count = 0
    if config.documentation.resources:
        resources = config.documentation.resources
        for category, items in resources.items():
            if isinstance(items, dict):
                resource_count += len(items)
            elif isinstance(items, list):
                resource_count += len(items)

    # Get vector DB status
    try:
        from langchain_chroma import Chroma
        from .embeddings import get_embeddings

        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=str(config.get_db_path()),
            embedding_function=embeddings
        )
        doc_count = vectorstore._collection.count()
        db_status = f"✅ {doc_count} documents"
    except Exception as e:
        doc_count = 0
        db_status = "❌ Not available"

    # Format summary
    summary = f"""
### 📊 Sources Overview

**Documentation Sources:**
- 📦 Git Repositories: {git_repo_count}
- 📁 Local Folders: {local_folder_count}
- 📖 Resource Entries: {resource_count}

**Vector Database:**
- Status: {db_status}
- Location: `{config.get_db_path()}`

**Total Sources Configured:** {git_repo_count + local_folder_count + resource_count}
"""
    return summary


def get_source_statuses():
    """
    Get detailed status for each configured source.
    Returns a formatted string with per-source status indicators.
    """
    config = get_config()
    statuses = []

    # Check git repos
    for repo_url in config.documentation.git_repos:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = Path(config.documentation.docs_output_dir) / repo_name

        if repo_path.exists():
            build_path = Path(config.documentation.sphinx_build_html_path)
            if build_path.exists():
                statuses.append(f"✅ {repo_url} (cloned & built)")
            else:
                statuses.append(f"⚠️ {repo_url} (cloned, not built)")
        else:
            statuses.append(f"❌ {repo_url} (not cloned)")

    # Check local folders
    for folder_path in config.documentation.local_folders:
        folder = Path(folder_path)
        if folder.exists() and folder.is_dir():
            statuses.append(f"✅ {folder_path} (found)")
        else:
            statuses.append(f"❌ {folder_path} (not found)")

    # Check resources
    try:
        if config.documentation.resources:
            resource_categories = len(config.documentation.resources)
            statuses.append(f"✅ Resources ({resource_categories} categories configured)")
    except:
        statuses.append("⚠️ Resources (could not verify)")

    if not statuses:
        return "No sources configured"

    return "\n".join(statuses)


def validate_resources_yaml(yaml_text):
    """
    Validate YAML syntax for resources.
    Returns validation status message.
    """
    import yaml

    try:
        # Try to parse the YAML
        parsed = yaml.safe_load(yaml_text)

        if not parsed:
            return "❌ Empty YAML"

        if not isinstance(parsed, dict):
            return "❌ YAML must be a dictionary/object"

        if 'resources' not in parsed:
            return "⚠️ Warning: No 'resources' key found at top level"

        # Check if resources is a dict
        resources = parsed['resources']
        if not isinstance(resources, dict):
            return "❌ 'resources' must be a dictionary"

        # Count categories
        category_count = len(resources)

        return f"✅ Valid YAML - {category_count} resource categories found"

    except yaml.YAMLError as e:
        return f"❌ Invalid YAML syntax: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def save_resources_config(yaml_text):
    """
    Save resources section to config.yaml.
    Validates YAML before saving.
    """
    import yaml

    try:
        # Validate first
        parsed = yaml.safe_load(yaml_text)

        if not parsed or not isinstance(parsed, dict):
            return "❌ Invalid YAML format"

        if 'resources' not in parsed:
            return "❌ YAML must contain 'resources' key at top level"

        # Load current config
        config = get_config()

        # Update just the resources section (now under documentation)
        resources = parsed['resources']

        # Save to config object
        config.documentation.resources = resources

        # Save to file
        save_config(config)

        # Count what was saved
        resource_count = 0
        for category, items in resources.items():
            if isinstance(items, dict):
                resource_count += len(items)
            elif isinstance(items, list):
                resource_count += len(items)

        return f"✅ Resources saved successfully! {len(resources)} categories, {resource_count} total entries. Re-ingest to update vector database."

    except yaml.YAMLError as e:
        return f"❌ YAML parsing error: {str(e)}"
    except Exception as e:
        return f"❌ Save failed: {str(e)}"


def update_llm_fields_from_provider(provider):
    """
    Update LLM-related fields based on selected provider.
    Returns: (model_choices, api_type, api_key_env, base_url_visibility, base_url_value)
    """
    # ANL Argo models available via OpenAI-compatible endpoint
    argo_models = [
        # OpenAI models
        "gpt4o", "gpt4olatest", "gpt4turbo", "gpt41", "gpt5", "gpt5mini",
        # Google models
        "gemini25pro", "gemini25flash",
        # Anthropic models
        "claudesonnet4", "claudesonnet45", "claudeopus4", "claudeopus45", "claudehaiku45",
    ]

    provider_config = {
        "gemini": {
            "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            "api_type": "google",
            "api_key_env": "GEMINI_API_KEY",
            "base_url": "",
        },
        "openai": {
            "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
            "api_type": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "",
        },
        "azure": {
            "models": ["gpt-4", "gpt-35-turbo"],
            "api_type": "azure",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "base_url": "",
        },
        "anthropic": {
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "api_type": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "",
        },
        "anl_argo": {
            "models": argo_models,
            "api_type": "openai",  # Argo uses OpenAI-compatible API
            "api_key_env": "",  # Use api_key directly in config for ANL Argo
            "base_url": "https://apps-dev.inside.anl.gov/argoapi/v1/",
        }
    }

    config = provider_config.get(provider, provider_config["gemini"])

    # Show base_url settings for ANL Argo
    base_url_visible = (provider == "anl_argo")

    return (
        gr.Dropdown(choices=config["models"], value=config["models"][0]),
        config["api_type"],
        config["api_key_env"],
        gr.Group(visible=base_url_visible),  # Show/hide base_url settings
        config["base_url"],  # Set base_url value
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
            gr.Markdown(
                "Click on a conversation title to load it. Use the buttons to download or delete conversations."
            )

            # HTML-based conversation list with inline buttons
            conversation_list_html = gr.HTML(generate_conversation_list_html())

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")

            # Hidden textboxes for JavaScript to interact with
            load_id_hidden = gr.Textbox(
                visible=False,
                placeholder="Conversation ID will be loaded here",
            )
            download_id_hidden = gr.Textbox(
                visible=False,
                placeholder="Conversation ID for download",
            )
            delete_id_hidden = gr.Textbox(
                visible=False,
                placeholder="Conversation ID for delete",
            )

            # Buttons triggered by JavaScript
            with gr.Row(visible=False):
                load_selected_btn = gr.Button("Load Selected")
                download_json_btn = gr.Button("Download JSON")
                delete_selected_btn = gr.Button("Delete Selected")

            # Status and output
            history_status = gr.Textbox(label="Status", interactive=False)

            # File output for downloads
            download_file = gr.File(label="Downloaded File", visible=True)

            # Conversation details panel
            with gr.Accordion("📄 Conversation Details", open=False):
                conversation_details = gr.Markdown(
                    "Click a conversation title to view details here."
                )

            # Connect functions
            refresh_btn.click(
                fn=generate_conversation_list_html,
                inputs=None,
                outputs=conversation_list_html,
            )

            # Load conversation when title is clicked (triggered by JavaScript)
            load_selected_btn.click(
                fn=load_conversation_and_refresh,
                inputs=load_id_hidden,
                outputs=[chatbot, conversation_details, history_status],
            )

            # Download conversation as JSON
            download_json_btn.click(
                fn=download_conversation_json,
                inputs=download_id_hidden,
                outputs=[download_file, history_status],
            )

            # Delete conversation
            delete_selected_btn.click(
                fn=delete_conversation_and_refresh,
                inputs=delete_id_hidden,
                outputs=[conversation_list_html, conversation_details, history_status],
            )

        # --- Tab 3: Sources ---
        with gr.Tab("Sources"):
            gr.Markdown("## Documentation Sources")
            gr.Markdown(
                "Manage documentation sources and run ingestion to update the vector database."
            )

            # Summary Panel
            sources_summary = gr.Markdown(value="Loading sources overview...")

            with gr.Row():
                summary_refresh_btn = gr.Button("🔄 Refresh Overview", size="sm", variant="secondary")

            with gr.Accordion("📚 Documentation Sources", open=True):
                gr.Markdown(
                    "*Configure where documentation should be loaded from. Git repositories will be "
                    "automatically cloned and built. Local folders should point to pre-built HTML documentation.*"
                )

                with gr.Row():
                    sources_refresh_btn = gr.Button("🔄 Refresh Values", size="sm", variant="secondary")

                sources_git_repos = gr.Textbox(
                    label="Git Repositories (one per line)",
                    lines=3,
                    placeholder="https://github.com/xray-imaging/2bm-docs.git",
                    info="Repository URLs will be cloned to the documentation output directory"
                )
                sources_local_folders = gr.Textbox(
                    label="Local Folders (one per line)",
                    lines=3,
                    placeholder="/path/to/built/docs",
                    info="Paths to pre-built HTML documentation directories"
                )
                sources_docs_output = gr.Textbox(
                    label="Documentation Output Directory",
                    info="Where cloned git repositories will be stored"
                )
                sources_sphinx_html = gr.Textbox(
                    label="Sphinx HTML Build Path",
                    info="Path to built Sphinx HTML files that will be indexed"
                )

                # Source Status Display
                gr.Markdown("### 📊 Source Status")
                sources_status_display = gr.Textbox(
                    label="Per-Source Status",
                    lines=5,
                    interactive=False,
                    value="Loading status..."
                )

            with gr.Accordion("🔄 Vector Database & Ingestion", open=True):
                sources_ingest_status = gr.Textbox(
                    label="Vector Database Status",
                    value="Checking...",
                    interactive=False,
                )

                with gr.Row():
                    save_sources_btn = gr.Button("💾 Save Sources", variant="primary")
                    reingest_btn = gr.Button("🔄 Re-Ingest", variant="secondary")

                sources_ingest_output = gr.Textbox(
                    label="Ingestion Log",
                    interactive=False,
                    lines=10,
                    placeholder="Ingestion output will appear here...",
                )

            with gr.Accordion("📖 Reference Resources", open=False):
                gr.Markdown(
                    "**Edit resources** like beamlines, software packages, and organizations. "
                    "These will be embedded in the vector database after re-ingestion."
                )
                gr.Markdown(
                    "*⚠️ Warning: Must be valid YAML format. Invalid YAML will cause errors.*"
                )

                sources_resources_editor = gr.Textbox(
                    label="Resources Configuration (YAML)",
                    lines=20,
                    interactive=True,
                    placeholder="resources:\n  beamlines:\n    2bm:\n      name: ...",
                    info="Edit YAML directly. Click 'Save Resources' to apply changes."
                )

                with gr.Row():
                    save_resources_btn = gr.Button("💾 Save Resources", variant="primary")
                    validate_resources_btn = gr.Button("✅ Validate YAML", size="sm", variant="secondary")

                resources_validation_output = gr.Textbox(
                    label="Validation Result",
                    interactive=False,
                    lines=2,
                    visible=True
                )

            sources_status = gr.Textbox(label="Status", interactive=False)

            # Load sources values on startup
            demo.load(
                load_sources_values,
                None,
                [
                    sources_git_repos,
                    sources_local_folders,
                    sources_docs_output,
                    sources_sphinx_html,
                    sources_resources_editor,
                ],
            )

            # Load initial summary and status
            demo.load(
                get_sources_summary,
                None,
                sources_summary
            )

            demo.load(
                get_source_statuses,
                None,
                sources_status_display
            )

            # Load initial ingestion status
            demo.load(
                check_vectordb_status,
                None,
                sources_ingest_status
            )

            # Connect summary refresh button
            summary_refresh_btn.click(
                get_sources_summary,
                None,
                sources_summary
            ).then(
                get_source_statuses,
                None,
                sources_status_display
            )

            # Connect source values refresh button
            sources_refresh_btn.click(
                load_sources_values,
                None,
                [
                    sources_git_repos,
                    sources_local_folders,
                    sources_docs_output,
                    sources_sphinx_html,
                    sources_resources_editor,
                ],
            ).then(
                get_source_statuses,
                None,
                sources_status_display
            )

            # Connect save sources button
            save_sources_btn.click(
                save_sources_values,
                [
                    sources_git_repos,
                    sources_local_folders,
                    sources_docs_output,
                    sources_sphinx_html,
                ],
                sources_status,
            )

            # Connect validate resources button
            validate_resources_btn.click(
                validate_resources_yaml,
                sources_resources_editor,
                resources_validation_output
            )

            # Connect save resources button
            save_resources_btn.click(
                save_resources_config,
                sources_resources_editor,
                resources_validation_output
            ).then(
                get_sources_summary,
                None,
                sources_summary
            )

            # Connect re-ingest button
            reingest_btn.click(
                run_data_ingestion,
                None,
                sources_ingest_output
            ).then(
                check_vectordb_status,
                None,
                sources_ingest_status
            ).then(
                get_sources_summary,
                None,
                sources_summary
            ).then(
                get_source_statuses,
                None,
                sources_status_display
            )

        # --- Tab 4: Configuration ---
        with gr.Tab("Configuration"):
            gr.Markdown("## Technical Configuration")
            gr.Markdown(
                "Edit RAG and LLM settings. **Restart the backend** to apply changes."
            )

            # Display config file path
            config_file_path = gr.Textbox(
                label="Configuration File Path",
                value=str(Path("config.yaml").absolute()),
                interactive=False,
                lines=1,
            )

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

                # Base URL settings (shown for ANL Argo and custom OpenAI-compatible endpoints)
                with gr.Group(visible=False) as base_url_settings:
                    gr.Markdown("### Custom API Endpoint")
                    gr.Markdown(
                        "*For ANL Argo, use: `https://apps-dev.inside.anl.gov/argoapi/v1/`*"
                    )
                    base_url = gr.Textbox(
                        label="Base URL",
                        placeholder="https://apps-dev.inside.anl.gov/argoapi/v1/",
                        value="",
                        info="Custom base URL for OpenAI-compatible APIs (leave empty for default)",
                    )

            with gr.Accordion("✂️ Text Processing", open=False):
                chunk_size = gr.Slider(
                    label="Chunk Size", minimum=100, maximum=5000, step=100
                )
                chunk_overlap = gr.Slider(
                    label="Chunk Overlap", minimum=0, maximum=1000, step=50
                )

            save_cfg_btn = gr.Button("💾 Save Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False)

            # Connect provider change event to update fields
            provider.change(
                update_llm_fields_from_provider,
                inputs=[provider],
                outputs=[llm_model, api_type, api_key_env, base_url_settings, base_url],
            )

            # Load current config values on startup for Configuration tab
            demo.load(
                load_config_values,
                None,
                [
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
                    base_url,
                    chunk_size,
                    chunk_overlap,
                ],
            )

            # Connect save config function
            save_cfg_btn.click(
                save_config_values,
                [
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
                    base_url,
                    chunk_size,
                    chunk_overlap,
                ],
                config_status,
            )

if __name__ == "__main__":
    demo.launch(
        server_name=config.server.frontend_host,
        server_port=config.server.frontend_port,
        allowed_paths=[DOCS_DIR],  # This is crucial for serving images
    )
