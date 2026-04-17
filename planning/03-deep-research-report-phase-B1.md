# Phase B Step One: Message Schema and MessagePool Design for Pub/Sub Agentic Orchestration

## Design goals and research grounding

Your base paper’s orchestration pattern (as summarized in secondary writeups of the paper) uses a **subscription-based message routing mechanism** where agents register for topics, publish to a shared pool, and retrieve only topic-matching messages to keep the system decoupled and modular. citeturn0search2turn4search2

For implementation, the cleanest “production-grade research prototype” pattern is **message-driven pub/sub** with **audit logs**, **correlation IDs**, and **topic scoping**. Microsoft’s multi-agent reference architecture recommends asynchronous message-driven workflows via a broker/event bus and explicitly calls out correlation IDs to link responses to requests. citeturn5search0

AutoGen’s low-level core concepts also formalize broadcast communication as **publish/subscribe with topics and subscriptions**, where publishers don’t need to know recipients and subscriptions control delivery. citeturn3search0turn3search1  
AutoGen further defines a topic as **(topic type, topic source)** and uses a string form like `Topic_Type/Topic_Source`. citeturn3search1

Your requirement set matches these principles:
- **In-memory list** for runtime speed
- **Persisted JSONL** (`logs/message_pool.jsonl`) for auditability
- **Topics** (prepare/train/evaluate)
- **Roles** (TaskManager, Prep, ModelManager, ModelDev)
- A schema that supports **commands, results, and errors**, and **correlation/run IDs**

The rest of this answer gives you the concrete **Message schema** + **MessagePool** implementation you can drop into `src/agents/` as the foundation for all Phase‑B agents.

## Message schema

### Topic convention

Use a two-part topic ID:
- `topic_type`: logical channel name (`prepare`, `train`, `evaluate`, `system`, `error`)
- `topic_source`: a scope key (e.g., `run_id` like `run_2026_04_14_001`), allowing isolation between runs

This mirrors AutoGen’s “Topic = (Topic Type, Topic Source)” design and its `type/source` string format. citeturn3search1

Examples:
- `prepare/run_2026_04_14_001`
- `train/run_2026_04_14_001`
- `evaluate/run_2026_04_14_001`

### Message fields

A minimal but future-proof schema:

- `schema_version`: `"v1"`
- `msg_id`: UUID4 string
- `seq`: monotonically increasing integer assigned by the pool
- `created_at`: ISO-8601 UTC string
- `topic`: `"type/source"` string
- `kind`: `"command" | "event" | "result" | "error"`
- `sender`: agent name/id (e.g., `"TaskManagerAgent"`)
- `sender_role`: role name (e.g., `"task_manager"`)
- `to`: optional list of recipient agent IDs (directed delivery)
- `to_roles`: optional list of recipient roles (role-directed delivery)
- `correlation_id`: optional ID tying replies/results to an original command (recommended for message-driven orchestration). citeturn5search0
- `run_id`: optional explicit run id (helps indexing without parsing topics)
- `payload`: JSON-serializable dict (your task config, paths, metrics, etc.)
- `tags`: optional list of strings
- `level`: `"info" | "warning" | "error"` (for audit, triage)

This schema is consistent with AutoGen’s stance that **messages are serializable data objects** (dataclass or Pydantic) and are the core means of agent communication. citeturn3search0

## MessagePool logic

### Core behaviors

1) **Publish**
- Validate topic + role strings
- Assign `seq`, `msg_id`, `created_at` if missing
- Append to in-memory list
- Append one JSON record to `logs/message_pool.jsonl` (auditability requirement)

2) **Subscribe**
- Agent registers interest in topic types (optionally scoped by source/run)
- The pool stores a cursor so the agent can poll only “new” messages

3) **Poll**
- Returns messages matching:
  - topic match (wildcards supported)
  - AND (if specified) direct recipient or recipient role match
- Advances the agent cursor (simple at-most-once delivery semantics for the runtime loop; the JSONL log preserves full history for replay)

4) **Replay/Inspect (optional but recommended)**
- Read a JSONL audit log and filter by topic/correlation_id/run_id to debug experiments (supports your thesis methodology and reproducibility)

## Drop-in directory layout for this step

Create:

```
src/
  agents/
    __init__.py
    messages.py
    message_pool.py
logs/
  (created on first run)
```

## Code: `src/agents/messages.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
import re
import uuid


MessageKind = Literal["command", "event", "result", "error"]
MessageLevel = Literal["info", "warning", "error"]

_TOPIC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*/[ -~]+$")  # simple "type/source" check
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TopicId:
    """
    Topic identifier formatted as 'type/source'.

    Inspired by publish/subscribe topic concepts used in agent runtimes where
    a topic is an indirection over agent IDs and can be represented as type/source. citeturn3search1
    """
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
    Messages are data-only objects and must be JSON-serializable. citeturn3search0
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

        # Basic JSON-serializability check (best-effort)
        # We do not fully serialize here to avoid overhead; MessagePool will handle JSON errors on persist.
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
```

## Code: `src/agents/message_pool.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable
import json
import threading

from .messages import Message, TopicId


@dataclass
class Subscription:
    agent_id: str
    agent_role: str
    topics: Set[str]               # topic types, e.g., {"prepare","train","evaluate"} or {"*"}
    topic_source: Optional[str]    # scope/run_id; None means any
    cursor: int                   # index into MessagePool._messages
    start_from: str               # "earliest" or "latest"


class MessagePool:
    """
    In-memory pub/sub message pool with JSONL persistence for auditability.

    - publish(): append to memory and to logs/message_pool.jsonl
    - subscribe(): register agent topics
    - poll(): retrieve messages for an agent since its cursor

    This matches the base paper's methodology goal of a shared pool and topic-based routing. citeturn0search2turn4search2
    It also aligns with agent-runtime broadcast pub/sub concepts: topics + subscriptions decouple publishers from recipients. citeturn3search1
    """

    def __init__(
        self,
        log_path: str = "logs/message_pool.jsonl",
        load_existing: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._messages: List[Message] = []
        self._seq: int = 0

        self._subs: Dict[str, Subscription] = {}

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if load_existing and self.log_path.exists():
            self._load_from_jsonl()

    def _load_from_jsonl(self) -> None:
        """
        Load messages from a prior run's JSONL log into memory.
        Useful to resume/debug. Does not restore subscriptions (agents re-subscribe).
        """
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msg = Message(**obj)
                self._messages.append(msg)
                self._seq = max(self._seq, msg.seq + 1)

    def size(self) -> int:
        return len(self._messages)

    def subscribe(
        self,
        agent_id: str,
        agent_role: str,
        topics: Iterable[str],
        *,
        topic_source: Optional[str] = None,
        start_from: str = "latest",  # "earliest"|"latest"
    ) -> None:
        if start_from not in ("earliest", "latest"):
            raise ValueError("start_from must be 'earliest' or 'latest'")

        with self._lock:
            cursor = 0 if start_from == "earliest" else len(self._messages)
            self._subs[agent_id] = Subscription(
                agent_id=agent_id,
                agent_role=agent_role,
                topics=set(topics),
                topic_source=topic_source,
                cursor=cursor,
                start_from=start_from,
            )

    def unsubscribe(self, agent_id: str) -> None:
        with self._lock:
            self._subs.pop(agent_id, None)

    def publish(self, msg: Message) -> Message:
        """
        Publish a message:
          - assigns msg_id, created_at, seq
          - validates schema
          - appends in-memory
          - appends to JSONL audit log
        """
        with self._lock:
            msg.ensure_ids()
            msg.seq = self._seq
            self._seq += 1
            msg.validate()

            self._messages.append(msg)
            self._append_jsonl(msg)
            return msg

    def _append_jsonl(self, msg: Message) -> None:
        record = msg.to_dict()
        # Append-only JSON Lines for auditability
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def poll(self, agent_id: str, *, max_messages: int = 50) -> List[Message]:
        """
        Return new messages since the agent's cursor that match its subscription.
        Advances cursor to end after scanning.
        """
        with self._lock:
            if agent_id not in self._subs:
                return []

            sub = self._subs[agent_id]
            start = sub.cursor
            end = len(self._messages)

            out: List[Message] = []
            for i in range(start, end):
                msg = self._messages[i]
                if self._matches(sub, msg):
                    out.append(msg)
                    if len(out) >= max_messages:
                        # Stop early but still advance cursor to current scan point + 1 to prevent re-reading.
                        sub.cursor = i + 1
                        return out

            # Advance cursor to end (all messages scanned)
            sub.cursor = end
            return out

    def _matches(self, sub: Subscription, msg: Message) -> bool:
        # Topic matching: message topic is "type/source"
        t = TopicId.parse(msg.topic)
        topic_type_ok = ("*" in sub.topics) or (t.topic_type in sub.topics)
        if not topic_type_ok:
            return False

        # Optional scope restriction
        if sub.topic_source is not None and t.source != sub.topic_source:
            return False

        # Directed delivery: if msg specifies recipients, enforce.
        if msg.to is not None and sub.agent_id not in msg.to:
            return False
        if msg.to_roles is not None and sub.agent_role not in msg.to_roles:
            return False

        # Prevent self-echo loops in pub/sub by default (optional).
        # AutoGen runtime similarly avoids delivering published messages back to the publisher. citeturn3search0
        if msg.sender == sub.agent_id:
            return False

        return True

    def get_all(self) -> List[Message]:
        """Debug helper: return a copy of all messages."""
        with self._lock:
            return list(self._messages)

    def find_by_correlation_id(self, correlation_id: str) -> List[Message]:
        with self._lock:
            return [m for m in self._messages if m.correlation_id == correlation_id]

    def find_by_topic(self, topic_type: str, topic_source: Optional[str] = None) -> List[Message]:
        with self._lock:
            out = []
            for m in self._messages:
                t = TopicId.parse(m.topic)
                if t.topic_type != topic_type:
                    continue
                if topic_source is not None and t.source != topic_source:
                    continue
                out.append(m)
            return out
```

## Quick sanity-check example

This is not required yet, but it’s helpful to validate that “agents can talk” before implementing agents:

```python
from src.agents.message_pool import MessagePool
from src.agents.messages import Message, TopicId

pool = MessagePool("logs/message_pool.jsonl", load_existing=False)

run_id = "run_2026_04_14_001"

pool.subscribe("PreparationAgent", "preparation", topics=["prepare"], topic_source=run_id, start_from="earliest")

# Task manager publishes a prepare command
m = Message.new(
    topic=TopicId("prepare", run_id),
    kind="command",
    sender="TaskManagerAgent",
    sender_role="task_manager",
    payload={"dataset_manifest": "data/processed/dataset_manifest.json"}
)
pool.publish(m)

msgs = pool.poll("PreparationAgent")
assert len(msgs) == 1
print(msgs[0].payload)
```

## Why this design will scale to your Phase B requirements

- It supports **pub/sub topics**, which aligns with agent-framework broadcast patterns (publish to topic, subscribers receive copies). citeturn3search1  
- It supports **role-directed messaging**, which gives you flexible routing without hard-wiring agent names—useful as you expand to more agents later.  
- It writes to **JSONL**, which is the simplest robust “append-only audit log” format (easy to diff, stream, replay).  
- It supports **correlation IDs** for command→result linking, a key feature in message-driven multi-agent orchestration. citeturn5search0  

When you’re ready for Step Two, we’ll implement:
- `BaseAgent` + the four specialized agents
- `orchestrator.py` message loop
- A minimal baseline model executor that publishes `evaluate` results (MAE/RMSE + plot path)
- Optuna skeleton inside `ModelManagerAgent` (interactive BO simulation)