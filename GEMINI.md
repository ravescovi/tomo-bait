# GEMINI.md

## Project Overview

This project, `tomo-bait`, is a RAG (Retrieval-Augmented Generation) system designed to answer questions about tomography beamline documentation at the Advanced Photon Source (APS). It uses AI-powered agents to provide answers based on ingested documentation.

The system is composed of:

*   **Project-Based Data Isolation:** Each project is defined in `config.yaml` with a `project.name` (e.g., "tomo"). All data (ChromaDB, conversations, documentation) is stored in `.bait-{name}/` directory.
*   **FastAPI Backend:** A Python backend that exposes a `/chat` endpoint. It uses `autogen` to manage a conversation between two AI agents:
    *   A `doc_expert` agent that uses an LLM (Gemini by default) to answer questions based on context.
    *   A `tool_worker` agent that retrieves relevant documentation using a tool.
*   **Gradio Frontend:** A modern web interface with four tabs:
    *   **Chat:** Conversational interface with image rendering support
    *   **History:** View and resume past conversations
    *   **Configuration:** Edit all settings with hot-reload
    *   **Setup:** AI-powered configuration generation
*   **Command-Line Interface (CLI):** A simple CLI for interacting with the chatbot from the terminal.
*   **Retriever:** A component that uses `langchain` and a `ChromaDB` vector store to find relevant documents. The embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.
*   **Documentation Sources:** The project can ingest from:
    *   Git repositories (cloned to `.bait-{name}/documentation/`)
    *   Local folders with pre-built documentation
    *   Resource definitions from `config.yaml` (beamlines, software packages, organizations)

## Building and Running

This project uses `pixi` for environment and task management.

**Prerequisites:**

*   `pixi` installed.
*   An API key for your chosen LLM provider (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) set in `.env` file.

**Key Commands (run from the project root):**

*   **Install dependencies:**
    ```bash
    pixi install
    ```
*   **Ingest documentation:**
    *This must be run before starting the application for the first time.*
    ```bash
    pixi run ingest
    ```
*   **Run the application (backend and frontend):**
    1.  Start the backend:
        ```bash
        pixi run start-backend
        ```
        The backend will be available at `http://127.0.0.1:8001`.
    2.  Start the frontend:
        ```bash
        pixi run start-frontend
        ```
        The frontend will be available at `http://127.0.0.1:8000`.

*   **Run the CLI:**
    ```bash
    pixi run run-cli "Your question here"
    ```

## Configuration

TomoBait uses a centralized `config.yaml` file with the following structure:

*   **project:** Define `name` (e.g., "tomo") and `data_dir` (e.g., ".bait-tomo")
*   **storage:** Conversations directory (defaults to `{data_dir}/conversations`)
*   **documentation:** Git repos, local folders, and resource definitions
    *   Resources include beamlines, software packages, organizations, etc.
*   **retriever:** ChromaDB path (defaults to `{data_dir}/chroma_db`), embedding model, search parameters
*   **llm:** LLM provider configuration (Gemini, OpenAI, Anthropic, Azure, ANL Argo)
*   **text_processing:** Chunk size and overlap settings
*   **server:** Backend and frontend host/port settings

All paths can be explicitly set or left null to use computed defaults based on `project.data_dir`.

## Directory Structure

When you run the ingestion process, TomoBait creates a project-specific directory:

```
.bait-tomo/
├── chroma_db/          # Vector database (embeddings)
├── conversations/      # Saved chat history
└── documentation/      # Cloned repositories and built Sphinx docs
```

This ensures all project data is isolated and easy to manage.

## Development Conventions

*   **Linting:** The project uses `ruff` for linting.
    ```bash
    pixi run lint
    ```
*   **Formatting:** The project uses `ruff` for formatting.
    ```bash
    pixi run format
    ```
*   **Coding Style:**
    *   Line length: 88 characters
    *   Indent width: 4 spaces
    *   Quote style: double quotes
*   **Source Code:** All source code is located in the `src` directory.
