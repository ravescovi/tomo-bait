"""
Centralized configuration management for TomoBait.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Configuration for project identity and base directories."""

    name: str = Field(
        default="tomo",
        description="Project identifier name (used in directory naming)",
    )
    data_dir: str = Field(
        default=".bait-tomo",
        description="Base directory for all project data",
    )


class StorageConfig(BaseModel):
    """Configuration for conversation and data storage."""

    conversations_dir: Optional[str] = Field(
        default=None,
        description="Directory for conversation storage (defaults to {data_dir}/conversations)",
    )


class DocumentationSourceConfig(BaseModel):
    """Configuration for documentation sources."""

    git_repos: List[str] = Field(
        default_factory=list,
        description="List of Git repository URLs to clone and index",
    )
    local_folders: List[str] = Field(
        default_factory=list, description="List of local folder paths to index"
    )
    docs_output_dir: Optional[str] = Field(
        default=None,
        description="Directory where documentation will be stored (defaults to {data_dir}/documentation)",
    )
    sphinx_build_html_path: Optional[str] = Field(
        default=None,
        description="Path to built Sphinx HTML documentation (defaults to {data_dir}/documentation/repos/*/docs/_build/html)",
    )
    resources: Optional[Dict] = Field(
        default=None,
        description="Reference resources (beamlines, software, organizations, etc.)",
    )


class RetrieverConfig(BaseModel):
    """Configuration for the document retriever."""

    db_path: Optional[str] = Field(
        default=None,
        description="ChromaDB persist directory (defaults to {data_dir}/chroma_db)",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model name",
    )
    k: int = Field(
        default=3, description="Number of documents to retrieve per query", ge=1, le=20
    )
    search_type: str = Field(
        default="similarity",
        description="Search type: similarity, mmr, or similarity_score_threshold",
    )
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum relevance score (for similarity_score_threshold)", ge=0.0, le=1.0
    )


class LLMConfig(BaseModel):
    """Configuration for the LLM and agents."""

    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable name containing the API key",
    )
    model: str = Field(
        default="gemini-2.5-flash", description="Model name (e.g., gemini-2.5-flash)"
    )
    api_type: str = Field(default="google", description="API type (google, openai, etc.)")
    system_message: str = Field(
        default=(
            "You are an expert on this project's documentation. "
            "A user will ask a question. Your 'query_documentation' tool "
            "will provide you with the *only* relevant context. "
            "**You must answer the user's question based *only* on that context.** "
            "If the context is not sufficient, say so. Do not make up answers."
        ),
        description="System message for the documentation expert agent",
    )

    # ANL Argo specific configuration
    anl_api_url: Optional[str] = Field(
        default=None,
        description="ANL Argo API endpoint URL (only for api_type='anl_argo')",
    )
    anl_user: Optional[str] = Field(
        default=None,
        description="ANL username for API requests (only for api_type='anl_argo')",
    )
    anl_model: Optional[str] = Field(
        default=None,
        description="ANL model name (only for api_type='anl_argo')",
    )


class TextProcessingConfig(BaseModel):
    """Configuration for document text processing."""

    chunk_size: int = Field(
        default=1000, description="Size of text chunks in characters", ge=100, le=5000
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between chunks in characters", ge=0, le=1000
    )


class ServerConfig(BaseModel):
    """Configuration for server settings."""

    backend_host: str = Field(default="127.0.0.1", description="Backend server host")
    backend_port: int = Field(default=8001, description="Backend server port")
    frontend_host: str = Field(default="0.0.0.0", description="Frontend server host")
    frontend_port: int = Field(default=8000, description="Frontend server port")


class TomoBaitConfig(BaseModel):
    """Main configuration for TomoBait application."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    documentation: DocumentationSourceConfig = Field(
        default_factory=DocumentationSourceConfig
    )
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def get_data_dir(self) -> Path:
        """Get the resolved data directory path."""
        return Path(self.project.data_dir)

    def get_conversations_dir(self) -> Path:
        """Get the resolved conversations directory path."""
        if self.storage.conversations_dir:
            return Path(self.storage.conversations_dir)
        return self.get_data_dir() / "conversations"

    def get_docs_output_dir(self) -> Path:
        """Get the resolved documentation output directory path."""
        if self.documentation.docs_output_dir:
            return Path(self.documentation.docs_output_dir)
        return self.get_data_dir() / "documentation"

    def get_sphinx_build_html_path(self) -> Optional[Path]:
        """Get the resolved Sphinx build HTML path."""
        if self.documentation.sphinx_build_html_path:
            return Path(self.documentation.sphinx_build_html_path)
        # Return None - let ingestion discover the path
        return None

    def get_db_path(self) -> Path:
        """Get the resolved ChromaDB path."""
        if self.retriever.db_path:
            return Path(self.retriever.db_path)
        return self.get_data_dir() / "chroma_db"

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        self.get_data_dir().mkdir(parents=True, exist_ok=True)
        self.get_conversations_dir().mkdir(parents=True, exist_ok=True)
        self.get_docs_output_dir().mkdir(parents=True, exist_ok=True)
        self.get_db_path().parent.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manager for loading and saving configuration."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[TomoBaitConfig] = None
        self._reload_callbacks = []

    def load(self) -> TomoBaitConfig:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                if config_dict:
                    self._config = TomoBaitConfig(**config_dict)
                else:
                    self._config = TomoBaitConfig()
        else:
            # Create default config if file doesn't exist
            self._config = TomoBaitConfig()
            self.save(self._config)

        # Ensure all required directories exist
        self._config.ensure_directories()

        return self._config

    def save(self, config: TomoBaitConfig) -> None:
        """Save configuration to YAML file."""
        self._config = config
        config_dict = config.model_dump()

        with open(self.config_path, "w") as f:
            yaml.safe_dump(
                config_dict, f, default_flow_style=False, sort_keys=False, indent=2
            )

    def backup_config(self) -> str:
        """
        Backup current config file with timestamp.
        Returns the backup file path.
        """
        import shutil
        from datetime import datetime

        if not self.config_path.exists():
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config_path.parent / f"config.yaml.backup.{timestamp}"

        shutil.copy2(self.config_path, backup_path)

        # Keep only last 5 backups
        backups = sorted(self.config_path.parent.glob("config.yaml.backup.*"))
        if len(backups) > 5:
            for old_backup in backups[:-5]:
                old_backup.unlink()

        return str(backup_path)

    def get(self) -> TomoBaitConfig:
        """Get current configuration (load if not already loaded)."""
        if self._config is None:
            return self.load()
        return self._config

    def reload(self) -> TomoBaitConfig:
        """Reload configuration from file and notify callbacks."""
        config = self.load()

        # Notify all registered callbacks
        for callback in self._reload_callbacks:
            try:
                callback(config)
            except Exception as e:
                print(f"Error in reload callback: {e}")

        return config

    def register_reload_callback(self, callback):
        """Register a callback to be called when config is reloaded."""
        self._reload_callbacks.append(callback)

    def unregister_reload_callback(self, callback):
        """Unregister a reload callback."""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)


# Global config manager instance
_config_manager = ConfigManager()


def get_config() -> TomoBaitConfig:
    """Get the current configuration."""
    return _config_manager.get()


def save_config(config: TomoBaitConfig) -> None:
    """Save the configuration."""
    _config_manager.save(config)


def reload_config() -> TomoBaitConfig:
    """Reload configuration from file."""
    return _config_manager.reload()


def backup_config() -> str:
    """Backup current config file."""
    return _config_manager.backup_config()


def register_reload_callback(callback):
    """Register a callback for config reload events."""
    _config_manager.register_reload_callback(callback)


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    return _config_manager
