# GEMINI.md

## Project Overview

This project, `tomo-bait`, is a chatbot designed to answer questions about the 2-BM beamline at the Advanced Photon Source (APS). It uses a RAG (Retrieval-Augmented Generation) architecture to provide answers based on the official 2-BM documentation.

The system is composed of:

*   **FastAPI Backend:** A Python backend that exposes a `/chat` endpoint. It uses `autogen` to manage a conversation between two AI agents:
    *   A `technician_agent` that uses a Gemini model to answer questions based on context.
    *   A `worker_agent` that retrieves relevant documentation using a tool.
*   **Gradio Frontend:** A web-based chat interface that interacts with the FastAPI backend. It is capable of displaying images from the documentation.
*   **Command-Line Interface (CLI):** A simple CLI for interacting with the chatbot from the terminal.
*   **Retriever:** A component that uses `langchain` and a `ChromaDB` vector store to find relevant documents from the 2-BM documentation. The embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.
*   **Documentation:** The project ingests documentation from the `tomo_documentation/2bm-docs` directory, which is the official documentation for the 2-BM beamline.

## Building and Running

This project uses `pixi` for environment and task management.

**Prerequisites:**

*   `pixi` installed.
*   A `GEMINI_API_KEY` environment variable set with a valid Gemini API key.

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
