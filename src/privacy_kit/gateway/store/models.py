"""SQLModel schema for the metadata-only audit store.

Hard invariant: **no column ever stores raw PII**. We record what kind of PII was
seen and how much — entity types and counts — plus operational metadata (tool,
model, tokens, timestamp). An encrypted raw-values table could be added later as a
separate, opt-in model; this schema deliberately leaves room for that without
changing these tables.
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
    model: str  # upstream model name the request targeted
    policy: str = "pseudonymize"
    language: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    entity_total: int = 0  # sum of distinct PII values across all types
    entity_counts: dict[str, int] = Field(default_factory=dict, sa_column=Column(JSON))

    detections: list["Detection"] = Relationship(back_populates="interaction")


class Detection(SQLModel, table=True):
    """Per-entity-type breakdown for an interaction (queryable; values-free)."""

    id: int | None = Field(default=None, primary_key=True)
    interaction_id: int = Field(foreign_key="interaction.id", index=True)
    entity_type: str = Field(index=True)
    count: int

    interaction: Interaction | None = Relationship(back_populates="detections")
