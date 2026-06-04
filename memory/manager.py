from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings as ChromaSettings
import json
import time

from config.settings import CHROMA_PERSIST_DIR
from config.providers import get_embeddings, collection_suffix

_CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)


class ShortTermMemory:
    """Conversation buffer for a single agent session."""

    def __init__(self, max_messages: int = 20):
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        if role == "human":
            self.messages.append(HumanMessage(content=content))
        elif role == "ai":
            self.messages.append(AIMessage(content=content))
        elif role == "system":
            self.messages.append(SystemMessage(content=content))

        if len(self.messages) > self.max_messages:
            # Trim by pairs to avoid orphaning an AI reply without its human message
            trim = self.messages[-self.max_messages:]
            if trim and isinstance(trim[0], AIMessage):
                trim = trim[1:]
            self.messages = trim

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def clear(self):
        self.messages = []

    def to_string(self) -> str:
        lines = []
        for msg in self.messages:
            role = type(msg).__name__.replace("Message", "")
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)


class LongTermMemory:
    """Persistent memory stored in vector DB."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.embeddings = get_embeddings()
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=_CHROMA_SETTINGS,
        )
        self.vectorstore = Chroma(
            collection_name=f"memory_{agent_id}{collection_suffix()}",
            embedding_function=self.embeddings,
            client=client,
        )

    def save(self, content: str, metadata: Dict[str, Any] = None):
        metadata = metadata or {}
        metadata["agent_id"] = self.agent_id
        metadata["timestamp"] = time.time()
        doc = Document(page_content=content, metadata=metadata)
        self.vectorstore.add_documents([doc])

    def recall(self, query: str, k: int = 3) -> List[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def recall_with_metadata(self, query: str, k: int = 3) -> List[Dict]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]


class AgentMemory:
    """Combined short + long term memory for an agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(agent_id)

    def has_memories(self) -> bool:
        """Return True only if there are stored long-term memories.
        Avoids hitting the embedding API on an empty collection."""
        try:
            return self.long_term.vectorstore._collection.count() > 0
        except Exception:
            return False

    def remember(self, content: str, important: bool = False):
        """Save to short term, optionally persist to long term."""
        self.short_term.add("ai", content)
        if important:
            self.long_term.save(content, {"type": "important"})

    def recall_relevant(self, query: str) -> str:
        """Get relevant memories for a query."""
        memories = self.long_term.recall(query, k=3)
        if not memories:
            return ""
        return "Relevant past memories:\n" + "\n---\n".join(memories)
