from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Literal, Optional
import uuid

MessageKind = Literal["command", "event", "result", "error"]
MessageLevel = Literal["info", "warning", "error"]

# Simple "type/source" convention:
# - topic type: identifier-ish (letters, numbers, _, -)
# - source: printable ASCII (to allow run IDs like "run_2026_04_14_001")
_TOPIC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*/[ -~]+$")
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TopicId:
    """Topic identifier formatted as 'type/source'."""

    topic_type: str
    source: str = "default"

    def as_str(self) -> str:
        return f"{self.topic_type}/{self.source}"

    @staticmethod
    def parse(topic: str) -> "TopicId":
        if "/" not in topic:
            raise ValueError(f"Topic must be 'type/source', got: {topic}")
        t, s = topic.split("/", 1)
        return TopicId(topic_type=t, source=s)


@dataclass
class Message:
    """
    Serializable message exchange format for agent communication.
    Messages must be JSON-serializable (payload is a dict).
    """

    schema_version: str = "v1"

    # Assigned by MessagePool.publish()
    msg_id: str = ""
    seq: int = -1
    created_at: str = ""

    topic: str = ""  # "type/source"

    kind: MessageKind = "event"
    level: MessageLevel = "info"

    sender: str = ""
    sender_role: str = ""

    # Optional directed delivery
    to: Optional[List[str]] = None
    to_roles: Optional[List[str]] = None

    # Correlation / run scope
    correlation_id: Optional[str] = None
    run_id: Optional[str] = None

    payload: Dict[str, Any] = field(default_factory=dict)
    tags: Optional[List[str]] = None

    def ensure_ids(self) -> None:
        if not self.msg_id:
            self.msg_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = utc_now_iso()

    def validate(self) -> None:
        if not self.topic or not _TOPIC_RE.match(self.topic):
            raise ValueError(f"Invalid topic '{self.topic}'. Expected 'type/source' with safe characters.")
        if self.sender and not _ID_RE.match(self.sender):
            raise ValueError(f"Invalid sender '{self.sender}'. Use [A-Za-z_][A-Za-z0-9_-]*.")
        if self.sender_role and not _ID_RE.match(self.sender_role):
            raise ValueError(f"Invalid sender_role '{self.sender_role}'. Use [A-Za-z_][A-Za-z0-9_-]*.")
        if self.to:
            for a in self.to:
                if not _ID_RE.match(a):
                    raise ValueError(f"Invalid recipient agent id '{a}'.")
        if self.to_roles:
            for r in self.to_roles:
                if not _ID_RE.match(r):
                    raise ValueError(f"Invalid recipient role '{r}'.")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dict (JSON-serializable).")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def new(
        *,
        topic: TopicId | str,
        kind: MessageKind,
        sender: str,
        sender_role: str,
        payload: Optional[Dict[str, Any]] = None,
        to: Optional[List[str]] = None,
        to_roles: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        run_id: Optional[str] = None,
        level: MessageLevel = "info",
        tags: Optional[List[str]] = None,
    ) -> "Message":
        topic_str = topic.as_str() if isinstance(topic, TopicId) else topic
        return Message(
            topic=topic_str,
            kind=kind,
            sender=sender,
            sender_role=sender_role,
            payload=payload or {},
            to=to,
            to_roles=to_roles,
            correlation_id=correlation_id,
            run_id=run_id,
            level=level,
            tags=tags,
        )

