# Phase B1 - Pub/Sub Message Schema + MessagePool (Foundation for Orchestration)

Phase B1 builds the communication backbone for later agentic orchestration. It does not train models. It defines:
- A strict, JSON-serializable `Message` schema
- An in-memory `MessagePool` pub/sub bus with append-only JSONL audit logging

Implemented files:
- `src/agents/messages.py`
- `src/agents/message_pool.py`

## What This Phase Enables

Agents can coordinate through topics without hard-coding who talks to whom:
- Task manager publishes `command`s to a topic (ex: `prepare/<run_id>`)
- Subscribed agents poll only the messages relevant to their topic scope and role
- Agents reply with `result` or `error` messages using the same `correlation_id`

## Inputs and Outputs

Inputs:
- `Message` objects created via `Message.new(...)`
- Agent subscriptions via `MessagePool.subscribe(...)`

Outputs:
- `MessagePool.poll(...)` returns the matching new messages for an agent
- `logs/message_pool.jsonl` stores every published message (JSON Lines) for audit/replay

## Core Schema (Message)

Topic convention:
- `topic = "type/source"` (example: `prepare/run_2026_04_16_001`)
  - `type`: logical channel (`prepare`, `train`, `evaluate`, `system`, `error`, etc.)
  - `source`: run scope (typically `run_id`) to isolate runs

Message kind:
- `command | event | result | error`

Tracing:
- `correlation_id`: links replies to the originating command
- `run_id`: optional explicit run id (topic source already scopes runs)

Directed delivery (optional):
- `to`: list of specific agent ids
- `to_roles`: list of recipient roles

Pool-assigned audit fields on publish:
- `msg_id` (UUID4), `seq` (monotonic int), `created_at` (UTC ISO-8601)

## Core Runtime Flow (Typical)

1. Create pool: `pool = MessagePool("logs/message_pool.jsonl", load_existing=...)`
2. Agents subscribe: `pool.subscribe(agent_id, agent_role, topics=[...], topic_source=run_id)`
3. Publisher posts a message: `pool.publish(Message.new(...))`
4. Subscribers receive via polling: `msgs = pool.poll(agent_id)`
5. Subscribers publish `result`/`error` with `correlation_id` to link back to the command

Delivery rules:
- Topic match by `topic_type` (or `*`)
- Optional scope match by `topic_source`
- Optional enforcement of `to` and/or `to_roles`
- No self-echo: a sender does not receive its own published messages

## What “Cursor” Means 

The “cursor” is internal state inside `MessagePool`:
- Each subscription stores `cursor = index into the pool’s in-memory message list`
- `poll(agent_id)` scans messages starting at `cursor`, returns matches, then moves `cursor` forward
- This prevents the same agent from receiving the same old messages again on the next poll (at-most-once delivery within a running process)

`start_from` behavior:
- `start_from="earliest"` sets `cursor=0` (receive all existing messages that match)
- `start_from="latest"` sets `cursor=len(messages)` (receive only new messages published after subscribing)

## Minimal Example

```python
from src.agents.message_pool import MessagePool
from src.agents.messages import Message, TopicId

pool = MessagePool("logs/message_pool.jsonl", load_existing=False)
run_id = "run_2026_04_16_001"

pool.subscribe("PreparationAgent", "preparation", topics=["prepare"], topic_source=run_id, start_from="earliest")

pool.publish(
    Message.new(
        topic=TopicId("prepare", run_id),
        kind="command",
        sender="TaskManagerAgent",
        sender_role="task_manager",
        payload={"dataset_manifest": "data/processed/dataset_manifest.json"},
    )
)

msgs = pool.poll("PreparationAgent")
assert len(msgs) == 1
print(msgs[0].payload)
```

## Verification (Working Status)

Run:

```bash
pytest -q
```

Test coverage includes `tests/test_message_pool_pubsub.py`:
- topic + source scoping
- directed delivery by role and agent id
- cursor advancement (no double-delivery)
- JSONL persistence + reload + correlation-id lookup
