"""
Conversation storage and management using JSON files.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Message content (text)")
    image_path: Optional[str] = Field(
        default=None, description="Optional image path for assistant messages"
    )
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Conversation(BaseModel):
    """A conversation with metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="New Conversation")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    messages: List[Message] = Field(default_factory=list)

    @property
    def message_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages)

    @property
    def preview(self) -> str:
        """Get a preview of the conversation (first user message)."""
        for msg in self.messages:
            if msg.role == "user":
                # Return first 100 characters of the first user message
                return msg.content[:100] + ("..." if len(msg.content) > 100 else "")
        return "No messages"

    def add_message(self, role: str, content: str, image_path: Optional[str] = None):
        """Add a message to the conversation."""
        msg = Message(role=role, content=content, image_path=image_path)
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()

    def to_gradio_history(self) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Convert messages to Gradio chatbot history format.
        Returns list of tuples: [(user_msg, assistant_msg), ...]
        """
        history = []
        for msg in self.messages:
            if msg.role == "user":
                history.append((msg.content, None))
            elif msg.role == "assistant":
                if msg.image_path:
                    history.append((None, msg.image_path))
                else:
                    history.append((msg.content, None))
        return history

    @classmethod
    def from_gradio_history(
        cls, history: List[Tuple[Optional[str], Optional[str]]], title: Optional[str] = None
    ) -> "Conversation":
        """
        Create a conversation from Gradio chatbot history format.
        """
        conv = cls(title=title or "New Conversation")
        for user_msg, assistant_msg in history:
            if user_msg:
                conv.add_message("user", user_msg)
            if assistant_msg:
                # Check if it's an image path or text
                if assistant_msg and (
                    assistant_msg.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))
                    or Path(assistant_msg).exists()
                ):
                    conv.add_message("assistant", "", image_path=assistant_msg)
                else:
                    conv.add_message("assistant", assistant_msg or "")
        return conv

    def generate_title(self) -> str:
        """Auto-generate a title from the first user message."""
        for msg in self.messages:
            if msg.role == "user":
                # Use first 50 characters
                title = msg.content[:50]
                if len(msg.content) > 50:
                    title += "..."
                return title
        return "Empty Conversation"


class ConversationStorage:
    """Manage conversation persistence using JSON files."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize conversation storage.

        Args:
            storage_dir: Path to storage directory. If None, will use config.
        """
        if storage_dir is None:
            # Import here to avoid circular dependency
            from .config import get_config
            config = get_config()
            self.storage_dir = config.get_conversations_dir()
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, conversation: Conversation) -> str:
        """Save a conversation to a JSON file. Returns conversation ID."""
        conversation.updated_at = datetime.now().isoformat()
        file_path = self.storage_dir / f"{conversation.id}.json"

        with open(file_path, "w") as f:
            json.dump(conversation.model_dump(), f, indent=2)

        return conversation.id

    def load(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation by ID."""
        file_path = self.storage_dir / f"{conversation_id}.json"

        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
            return Conversation(**data)

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation by ID. Returns True if deleted."""
        file_path = self.storage_dir / f"{conversation_id}.json"

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_all(self) -> List[Dict]:
        """
        List all conversations with metadata (no messages).
        Returns list sorted by updated_at (most recent first).
        """
        conversations = []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Only include metadata, not full messages
                    conversations.append(
                        {
                            "id": data["id"],
                            "title": data["title"],
                            "created_at": data["created_at"],
                            "updated_at": data["updated_at"],
                            "message_count": len(data.get("messages", [])),
                            "preview": self._get_preview(data.get("messages", [])),
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted files
                continue

        # Sort by updated_at, most recent first
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations

    def _get_preview(self, messages: List[Dict]) -> str:
        """Get preview text from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return content[:100] + ("..." if len(content) > 100 else "")
        return "No messages"


# Global storage instance
_storage = ConversationStorage()


def get_storage() -> ConversationStorage:
    """Get the global storage instance."""
    return _storage
