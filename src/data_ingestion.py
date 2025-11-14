"""
A module for cloning and updating a git repository, and then building its
Sphinx documentation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Union
import os

from git import Repo
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_documentation(repo_url: str, documentation_dir: Union[str, Path]):
    """
    Clones a repository if it doesn't exist, or pulls the latest changes if it does.
    Then, it builds the Sphinx documentation.

    Args:
        repo_url (str): The URL of the git repository to clone.
        documentation_dir (Union[str, Path]): The path to the directory where the
            documentation and repository will be stored.
    """
    documentation_dir = Path(documentation_dir)
    repo_dir = documentation_dir / "2bm-docs"

    # --- 1. Clone or Pull Repository ---
    if not repo_dir.exists():
        print(f"Cloning repository to: {repo_dir}")
        try:
            Repo.clone_from(repo_url, repo_dir)
        except Exception as e:
            print(f"❌ ERROR: Cloning failed: {e}")
            sys.exit(1)
    else:
        print(f"Pulling latest changes in repository: {repo_dir}")
        try:
            repo = Repo(repo_dir)
            origin = repo.remotes.origin
            origin.pull()
        except Exception as e:
            print(f"❌ ERROR: Pulling failed: {e}")
            sys.exit(1)

    # --- 2. Build Sphinx Documentation ---
    docs_path = repo_dir / "docs"
    if not docs_path.exists():
        print(f"❌ ERROR: 'docs' directory not found in repository: {docs_path}")
        sys.exit(1)

    # It's better to run sphinx-build from the original working directory
    # and specify the source and output directories.
    # This avoids issues with `os.chdir`.
    output_dir = docs_path / "_build"
    command = [
        "sphinx-build",
        "-b",
        "html",  # Build HTML
        str(docs_path),  # Source directory
        str(output_dir),  # Output directory
    ]

    print(f"Running Sphinx build: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Sphinx build successful. Output in: {output_dir}")

    except FileNotFoundError:
        print("❌ ERROR: 'sphinx-build' command not found.")
        print("Please make sure Sphinx is installed in your Python environment.")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Sphinx build failed with code {e.returncode}.")
        print("\n--- Sphinx Output (stdout) ---")
        print(e.stdout)
        print("\n--- Sphinx Errors (stderr) ---")
        print(e.stderr)
        sys.exit(1)

def load_chunk_embed(HTML_BUILD_DIR: str):

    print(f"Loading docs from {HTML_BUILD_DIR}...")
    loader = ReadTheDocsLoader(HTML_BUILD_DIR)
    docs = loader.load()

    if not docs:
        print("❌ ERROR: No documents were loaded. Check your HTML_BUILD_DIR.")
        sys.exit(1)

    print(f"✅ Loaded {len(docs)} documents.")

    # This splitter tries to keep paragraphs/sentences together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The size of each chunk in characters
        chunk_overlap=200   # How much chunks overlap
    )

    print("Splitting documents into chunks...")
    splits = text_splitter.split_documents(docs)
    print(f"✅ Split {len(docs)} docs into {len(splits)} chunks.")

    # Define where to save the database
    DB_PATH = "./chroma_db"

    print("Initializing embedding model3...")
    # This model will be downloaded and run 100% locally
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("✅ Using local, open-source embeddings!")
    print(f"Creating and saving vector store at {DB_PATH}...")
    # This is the magic command.
    # It takes all splits, embeds them, and saves to disk.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
)

    print("🎉 All done!")
    print(f"Your knowledge base is ready and saved in '{DB_PATH}'.")

if __name__ == "__main__":
    current_directory =  Path.cwd()

    REPO_URL = "https://github.com/xray-imaging/2bm-docs.git"
    # Store documentation in a 'tomo_documentation' folder in the user's home directory
    DOCS_DIR = current_directory / "tomo_documentation"
    ingest_documentation(REPO_URL, DOCS_DIR)

    HTML_BUILD_DIR = DOCS_DIR / "2bm-docs/docs/_build/html"
    load_chunk_embed(HTML_BUILD_DIR)
