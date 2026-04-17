from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .message_pool import MessagePool
from .messages import Message, TopicId


@dataclass
class AgentConfig:
    agent_id: str
    role: str
    run_id: str


class BaseAgent:
    """
    Base class for agents communicating via MessagePool.
    Agents subscribe to topic types for a run_id scope (topic source), poll on heartbeat ticks,
    and publish commands/events/results/errors.
    """

    def __init__(self, cfg: AgentConfig, pool: MessagePool):
        self.agent_id = cfg.agent_id
        self.role = cfg.role
        self.run_id = cfg.run_id
        self.pool = pool

    def subscribe(self, topics: Iterable[str], *, start_from: str = "latest") -> None:
        self.pool.subscribe(
            agent_id=self.agent_id,
            agent_role=self.role,
            topics=set(topics),
            topic_source=self.run_id,
            start_from=start_from,
        )

    def poll(self, *, max_messages: int = 50) -> List[Message]:
        return self.pool.poll(self.agent_id, max_messages=max_messages)

    def publish(
        self,
        *,
        topic_type: str,
        kind: str,
        payload: dict,
        correlation_id: Optional[str] = None,
        level: str = "info",
        to: Optional[list[str]] = None,
        to_roles: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Message:
        msg = Message.new(
            topic=TopicId(topic_type, self.run_id),
            kind=kind,
            sender=self.agent_id,
            sender_role=self.role,
            payload=payload,
            correlation_id=correlation_id,
            run_id=self.run_id,
            level=level,
            to=to,
            to_roles=to_roles,
            tags=tags,
        )
        return self.pool.publish(msg)

    def tick(self) -> None:
        raise NotImplementedError

