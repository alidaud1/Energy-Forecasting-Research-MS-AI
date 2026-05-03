from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Set

from .messages import Message, TopicId


# Process-wide lock to prevent concurrent JSONL writes from multiple MessagePool instances
# (e.g., multiple runs in the WorkbenchEngine) corrupting logs/message_pool.jsonl.
_JSONL_WRITE_LOCK = threading.Lock()


@dataclass
class Subscription:
    agent_id: str
    agent_role: str
    topics: Set[str]  # topic types (e.g. {"prepare","train"} or {"*"})
    topic_source: Optional[str]  # scope/run_id; None means any
    cursor: int  # index into MessagePool._messages
    start_from: str  # "earliest" | "latest"


class MessagePool:
    """
    In-memory pub/sub message pool with JSONL persistence for auditability.

    Behaviors:
    - publish(): assign ids + seq, validate, append to memory + logs/message_pool.jsonl
    - subscribe(): register an agent subscription (topics + optional source scope)
    - poll(): retrieve messages since cursor that match subscription and advance cursor
    """

    def __init__(self, log_path: str = "logs/message_pool.jsonl", load_existing: bool = True) -> None:
        self._lock = threading.Lock()
        self._messages: List[Message] = []
        self._seq: int = 0
        self._subs: Dict[str, Subscription] = {}

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if load_existing and self.log_path.exists():
            self._load_from_jsonl()

    def _load_from_jsonl(self) -> None:
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msg = Message(**obj)
                self._messages.append(msg)
                if msg.seq >= 0:
                    self._seq = max(self._seq, msg.seq + 1)

    def size(self) -> int:
        with self._lock:
            return len(self._messages)

    def subscribe(
        self,
        agent_id: str,
        agent_role: str,
        topics: Iterable[str],
        *,
        topic_source: Optional[str] = None,
        start_from: str = "latest",
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
        # Multiple MessagePool instances may share the same log_path; guard the append
        # with a global lock to keep each JSON object on a single line.
        with _JSONL_WRITE_LOCK:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def cursor_from_seq(self, after_seq: int) -> int:
        """
        Translate an SSE/WS last_seq into an internal list cursor index.
        Optimized for the common case where after_seq is near the end.
        """
        with self._lock:
            if after_seq < 0:
                return 0
            i = len(self._messages) - 1
            while i >= 0 and int(self._messages[i].seq) > int(after_seq):
                i -= 1
            return i + 1

    def read_from_cursor(
        self,
        cursor: int,
        *,
        run_id: Optional[str] = None,
        max_messages: int = 500,
    ) -> tuple[List[Message], int]:
        """
        Cursor-based read for API streaming (SSE/WS).
        Returns (messages, new_cursor). The cursor is an index into the internal message list.
        """
        with self._lock:
            start = max(0, int(cursor))
            end = len(self._messages)
            out: List[Message] = []
            i = start
            while i < end:
                m = self._messages[i]
                if run_id is None or str(m.run_id or "") == str(run_id):
                    out.append(m)
                    if len(out) >= int(max_messages):
                        return out, i + 1
                i += 1
            return out, end

    def poll(self, agent_id: str, *, max_messages: int = 50) -> List[Message]:
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
                        sub.cursor = i + 1
                        return out

            sub.cursor = end
            return out

    def _matches(self, sub: Subscription, msg: Message) -> bool:
        t = TopicId.parse(msg.topic)
        if not (("*" in sub.topics) or (t.topic_type in sub.topics)):
            return False

        if sub.topic_source is not None and t.source != sub.topic_source:
            return False

        # Directed delivery
        if msg.to is not None and sub.agent_id not in msg.to:
            return False
        if msg.to_roles is not None and sub.agent_role not in msg.to_roles:
            return False

        # Avoid self-echo by default
        if msg.sender and msg.sender == sub.agent_id:
            return False

        return True

    def get_all(self) -> List[Message]:
        with self._lock:
            return list(self._messages)

    def find_by_correlation_id(self, correlation_id: str) -> List[Message]:
        with self._lock:
            return [m for m in self._messages if m.correlation_id == correlation_id]

    def find_by_topic(self, topic_type: str, topic_source: Optional[str] = None) -> List[Message]:
        with self._lock:
            out: List[Message] = []
            for m in self._messages:
                t = TopicId.parse(m.topic)
                if t.topic_type != topic_type:
                    continue
                if topic_source is not None and t.source != topic_source:
                    continue
                out.append(m)
            return out
