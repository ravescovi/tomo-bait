"""
A module for cloning and updating a git repository, and then building its
Sphinx documentation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Union, Dict, Any, List

from git import Repo
from langchain_chroma import Chroma
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_config

# Load configuration
config = get_config()


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
        chunk_size=config.text_processing.chunk_size,
        chunk_overlap=config.text_processing.chunk_overlap
    )

    print("Splitting documents into chunks...")
    splits = text_splitter.split_documents(docs)
    print(f"✅ Split {len(docs)} docs into {len(splits)} chunks.")

    print("Initializing embedding model...")
    # This model will be downloaded and run 100% locally
    embeddings = HuggingFaceEmbeddings(model_name=config.retriever.embedding_model)

    print("✅ Using local, open-source embeddings!")
    print(f"Creating and saving vector store at {config.retriever.db_path}...")
    # This is the magic command.
    # It takes all splits, embeds them, and saves to disk.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=config.retriever.db_path
    )

    print("🎉 All done!")
    print(f"Your knowledge base is ready and saved in '{config.retriever.db_path}'.")


def create_resource_documents() -> List[Document]:
    """
    Convert resources from config.yaml into LangChain documents for embedding.
    This makes all the beamline info, software packages, etc. searchable.
    """
    documents = []

    # Get resources from config
    config_dict = config.model_dump()
    resources = config_dict.get("resources", {})

    if not resources:
        print("⚠️  No resources found in config.yaml")
        return documents

    print(f"📚 Creating documents from {len(resources)} resource categories...")

    # Helper function to recursively extract information
    def dict_to_text(data: Dict[str, Any], prefix: str = "") -> str:
        """Convert a nested dictionary to readable text."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(dict_to_text(value, prefix + "  "))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(dict_to_text(item, prefix + "  - "))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    # Process beamlines
    if "beamlines" in resources:
        for beamline_id, beamline_info in resources["beamlines"].items():
            content = f"Beamline: {beamline_id.upper()}\n\n"
            content += dict_to_text(beamline_info)

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "config_resources",
                    "category": "beamline",
                    "beamline_id": beamline_id
                }
            ))

    # Process organizations
    if "organizations" in resources:
        for org_id, org_info in resources["organizations"].items():
            content = f"Organization: {org_id.upper()}\n\n"
            content += dict_to_text(org_info)

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "config_resources",
                    "category": "organization",
                    "organization_id": org_id
                }
            ))

    # Process software categories
    if "software" in resources:
        for software_category, packages in resources["software"].items():
            for package_name, package_info in packages.items():
                content = f"Software Package: {package_name}\n"
                content += f"Category: {software_category}\n\n"
                content += dict_to_text(package_info)

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": "config_resources",
                        "category": "software",
                        "software_category": software_category,
                        "package_name": package_name
                    }
                ))

    # Process python ecosystem
    if "python_ecosystem" in resources:
        for eco_category, packages in resources["python_ecosystem"].items():
            for package in packages:
                if isinstance(package, dict):
                    content = f"Python Package: {package.get('name', 'Unknown')}\n"
                    content += f"Ecosystem Category: {eco_category}\n\n"
                    content += dict_to_text(package)

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": "config_resources",
                            "category": "python_ecosystem",
                            "ecosystem_category": eco_category,
                            "package_name": package.get("name", "Unknown")
                        }
                    ))

    # Process community resources
    if "community" in resources:
        for community_category, items in resources["community"].items():
            for item in items:
                if isinstance(item, dict):
                    content = f"Community Resource\n"
                    content += f"Category: {community_category}\n\n"
                    content += dict_to_text(item)

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": "config_resources",
                            "category": "community",
                            "community_category": community_category
                        }
                    ))

    # Process GitHub organizations
    if "github_organizations" in resources:
        for org in resources["github_organizations"]:
            if isinstance(org, dict):
                content = f"GitHub Organization: {org.get('name', 'Unknown')}\n\n"
                content += dict_to_text(org)

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": "config_resources",
                        "category": "github_organization",
                        "organization_name": org.get("name", "Unknown")
                    }
                ))

    print(f"✅ Created {len(documents)} resource documents")
    return documents


def embed_resources():
    """
    Embed resource documents from config.yaml into the vector store.
    """
    print("\n🔧 Processing resources from config.yaml...")

    # Create documents from resources
    resource_docs = create_resource_documents()

    if not resource_docs:
        print("⚠️  No resource documents to embed")
        return

    # Initialize embeddings
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=config.retriever.embedding_model)

    # Load existing vectorstore or create new one
    print(f"Adding resource documents to vector store at {config.retriever.db_path}...")

    # Check if vectorstore exists
    db_path = Path(config.retriever.db_path)
    if db_path.exists():
        # Add to existing vectorstore
        vectorstore = Chroma(
            persist_directory=config.retriever.db_path,
            embedding_function=embeddings
        )
        vectorstore.add_documents(resource_docs)
        print(f"✅ Added {len(resource_docs)} resource documents to existing vector store")
    else:
        # Create new vectorstore with resource docs
        vectorstore = Chroma.from_documents(
            documents=resource_docs,
            embedding=embeddings,
            persist_directory=config.retriever.db_path
        )
        print(f"✅ Created new vector store with {len(resource_docs)} resource documents")

    print("🎉 Resources embedded successfully!")

if __name__ == "__main__":
    print("🚀 Starting data ingestion process...")
    print(f"Configuration loaded from config.yaml")

    # Process all git repositories
    for repo_url in config.documentation.git_repos:
        print(f"\n📦 Processing repository: {repo_url}")
        ingest_documentation(repo_url, config.documentation.docs_output_dir)

    # Process all local folders
    for local_folder in config.documentation.local_folders:
        print(f"\n📁 Processing local folder: {local_folder}")
        # Local folders are already built, just load and embed
        if Path(local_folder).exists():
            load_chunk_embed(local_folder)
        else:
            print(f"⚠️  WARNING: Local folder does not exist: {local_folder}")

    # Load, chunk, and embed from the built HTML
    print(f"\n📚 Loading and embedding documentation from: {config.documentation.sphinx_build_html_path}")
    load_chunk_embed(config.documentation.sphinx_build_html_path)

    # Embed resources from config.yaml
    embed_resources()
