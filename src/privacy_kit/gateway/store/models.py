"""SQLModel schema for the audit store.

``Interaction`` and ``Detection`` are values-free metadata: what kind of PII was
seen and how much — entity types and counts — plus operational metadata (tool,
model, tokens, timestamp). ``InteractionText`` is the raw-values table: it stores
user-authored text and tool/file data segments (original + anonymized) in
plaintext.  System prompts, instruction blocks, tool-call arguments, and
assistant turns are never eligible for storage.  Among eligible segments the
subset that actually lands here is further governed by ``Settings.save_texts``.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Interaction(SQLModel, table=True):
    """One processed prompt/response that passed through the gateway."""

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow, index=True)
    source: str = Field(index=True)  # e.g. "claude-code", "codex", "cursor", "otel"
    wire_format: str  # "anthropic" | "openai_chat" | "openai_responses" | "otel"
    kind: str = Field(default="main", index=True)  # purpose: "main" | "safety" | "helper"
    model: str  # upstream model name the request targeted
    policy: str = "pseudonymize"
    language: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    entity_total: int = 0  # sum of distinct PII values across all types
    entity_counts: dict[str, int] = Field(default_factory=dict, sa_column=Column(JSON))

    detections: list["Detection"] = Relationship(back_populates="interaction")
    texts: list["InteractionText"] = Relationship(back_populates="interaction")


class Detection(SQLModel, table=True):
    """Per-entity-type breakdown for an interaction (queryable; values-free)."""

    id: int | None = Field(default=None, primary_key=True)
    interaction_id: int = Field(foreign_key="interaction.id", index=True)
    entity_type: str = Field(index=True)
    count: int

    interaction: Interaction | None = Relationship(back_populates="detections")


class InteractionText(SQLModel, table=True):
    """One user-authored or tool/file-data segment for an interaction.

    Unlike Interaction/Detection this stores raw text in plaintext.  Only
    segments routed through a USER/TOOL author in the request transforms are
    eligible (user messages and tool/function outputs); system prompts,
    instruction blocks, tool-call arguments, and assistant turns are excluded
    at source.  Among eligible segments, which ones actually land here is
    further governed by ``Settings.save_texts``.  Model output is never stored.
    """

    id: int | None = Field(default=None, primary_key=True)
    interaction_id: int = Field(foreign_key="interaction.id", index=True)
    seq: int = 0  # capture order within the request
    original: str
    anonymized: str

    interaction: Interaction | None = Relationship(back_populates="texts")
