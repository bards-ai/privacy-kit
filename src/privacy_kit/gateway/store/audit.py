"""Audit store repository.

Thin wrapper over a SQLite engine for recording interactions and reading back
aggregates. Construct one :class:`AuditStore` per process and reuse it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlmodel import Session, SQLModel, create_engine, func, select

from privacy_kit.gateway.config import get_settings
from privacy_kit.gateway.store.models import Detection, Interaction


class AuditStore:
    """Records and queries metadata-only audit rows."""

    def __init__(self, db_path: Path | str | None = None, *, echo: bool = False) -> None:
        path = str(db_path or get_settings().db_path)
        url = path if path.startswith("sqlite") else f"sqlite:///{path}"
        self.engine = create_engine(url, echo=echo)
        SQLModel.metadata.create_all(self.engine)

    def record(
        self,
        *,
        source: str,
        wire_format: str,
        model: str,
        entity_counts: Mapping[str, int],
        policy: str = "pseudonymize",
        language: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> int:
        """Persist one interaction and its per-type detections; return its id."""
        counts = {etype: int(n) for etype, n in entity_counts.items() if n}
        interaction = Interaction(
            source=source,
            wire_format=wire_format,
            model=model,
            policy=policy,
            language=language,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            entity_total=sum(counts.values()),
            entity_counts=counts,
        )
        with Session(self.engine) as session:
            session.add(interaction)
            session.flush()  # assigns interaction.id
            assert interaction.id is not None
            for etype, count in counts.items():
                session.add(
                    Detection(interaction_id=interaction.id, entity_type=etype, count=count)
                )
            session.commit()
            return interaction.id

    def summary(self) -> dict[str, Any]:
        """Aggregate stats across all recorded interactions."""
        with Session(self.engine) as session:
            interactions = session.exec(select(func.count()).select_from(Interaction)).one()
            rows = session.exec(
                select(Detection.entity_type, func.sum(Detection.count)).group_by(
                    Detection.entity_type
                )
            ).all()
        by_type = {etype: int(total) for etype, total in rows}
        return {
            "interactions": int(interactions),
            "entities_total": sum(by_type.values()),
            "entities_by_type": by_type,
        }

    def recent(self, limit: int = 50) -> list[Interaction]:
        """Return the most recent interactions, newest first."""
        with Session(self.engine) as session:
            statement = select(Interaction).order_by(Interaction.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
            return list(session.exec(statement).all())
