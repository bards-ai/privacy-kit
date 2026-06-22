"""Audit store repository.

Thin wrapper over a SQLite engine for recording interactions and reading them
back — both the simple aggregates the legacy ``/ui`` needs and the richer
filtered, paginated queries the dashboard API (`/api/v1`) serves. Construct one
:class:`AuditStore` per process and reuse it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import ColumnElement, delete, event
from sqlmodel import Session, SQLModel, col, create_engine, func, select

from privacy_kit.gateway.config import get_settings
from privacy_kit.gateway.store.models import Detection, Interaction, InteractionText

# Columns the list endpoint is allowed to sort by (anything else falls back to
# created_at), keyed by the name the API accepts.
_SORT_COLUMNS = {
    "created_at": Interaction.created_at,
    "source": Interaction.source,
    "wire_format": Interaction.wire_format,
    "model": Interaction.model,
    "policy": Interaction.policy,
    "language": Interaction.language,
    "entity_total": Interaction.entity_total,
    "input_tokens": Interaction.input_tokens,
    "output_tokens": Interaction.output_tokens,
    "id": Interaction.id,
}


class AuditStore:
    """Records and queries audit rows: metadata plus saved text segments."""

    def __init__(self, db_path: Path | str | None = None, *, echo: bool = False) -> None:
        path = str(db_path or get_settings().db_path)
        url = path if path.startswith("sqlite") else f"sqlite:///{path}"
        # check_same_thread=False: the proxy writes from the event loop while the
        # dashboard may read from a threadpool worker; sessions are short-lived
        # and never shared, so cross-thread connection reuse is safe here.
        self.engine = create_engine(url, echo=echo, connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:
            # WAL lets dashboard reads run concurrently with proxy writes; the
            # busy timeout avoids spurious "database is locked" under contention.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

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
        texts: Sequence[tuple[str, str]] = (),
    ) -> int:
        """Persist one interaction, its per-type detections, and any saved text
        segments (``(original, anonymized)`` pairs, in capture order); return its id.
        """
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
            for seq, (original, anonymized) in enumerate(texts):
                session.add(
                    InteractionText(
                        interaction_id=interaction.id,
                        seq=seq,
                        original=original,
                        anonymized=anonymized,
                    )
                )
            session.commit()
            return interaction.id

    # --- Legacy aggregates (used by /ui and `privacy-kit report`) -----------

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

    def texts(self, interaction_id: int) -> list[InteractionText]:
        """Saved text segments for one interaction, in capture order."""
        with Session(self.engine) as session:
            statement = (
                select(InteractionText)
                .where(InteractionText.interaction_id == interaction_id)
                .order_by(col(InteractionText.seq))
            )
            return list(session.exec(statement).all())

    def recent(self, limit: int = 50) -> list[Interaction]:
        """Return the most recent interactions, newest first."""
        with Session(self.engine) as session:
            statement = (
                select(Interaction).order_by(col(Interaction.created_at).desc()).limit(limit)
            )
            return list(session.exec(statement).all())

    # --- Dashboard queries (used by /api/v1) -------------------------------

    def _conditions(
        self,
        *,
        sources: Sequence[str] | None = None,
        wire_formats: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        policies: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        entity_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_entities: int | None = None,
        q: str | None = None,
    ) -> list[ColumnElement[bool]]:
        """Build the WHERE clauses shared by the list, count, and export paths."""
        conds: list[ColumnElement[bool]] = []
        if sources:
            conds.append(col(Interaction.source).in_(sources))
        if wire_formats:
            conds.append(col(Interaction.wire_format).in_(wire_formats))
        if models:
            conds.append(col(Interaction.model).in_(models))
        if policies:
            conds.append(col(Interaction.policy).in_(policies))
        if languages:
            conds.append(col(Interaction.language).in_(languages))
        if min_entities is not None:
            conds.append(col(Interaction.entity_total) >= min_entities)
        if date_from is not None:
            conds.append(col(Interaction.created_at) >= date_from)
        if date_to is not None:
            conds.append(col(Interaction.created_at) <= date_to)
        if entity_type:
            conds.append(
                select(Detection.id)
                .where(
                    col(Detection.interaction_id) == col(Interaction.id),
                    col(Detection.entity_type) == entity_type,
                )
                .exists()
            )
        if q:
            like = f"%{q}%"
            conds.append(
                select(InteractionText.id)
                .where(
                    col(InteractionText.interaction_id) == col(Interaction.id),
                    col(InteractionText.original).like(like)
                    | col(InteractionText.anonymized).like(like),
                )
                .exists()
            )
        return conds

    def query_interactions(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        sort: str = "created_at",
        order: str = "desc",
        sources: Sequence[str] | None = None,
        wire_formats: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        policies: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        entity_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_entities: int | None = None,
        q: str | None = None,
    ) -> tuple[list[Interaction], int]:
        """One page of interactions matching the filters, plus the total count."""
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        column = _SORT_COLUMNS.get(sort, Interaction.created_at)
        direction = col(column).asc() if order == "asc" else col(column).desc()
        conds = self._conditions(
            sources=sources,
            wire_formats=wire_formats,
            models=models,
            policies=policies,
            languages=languages,
            entity_type=entity_type,
            date_from=date_from,
            date_to=date_to,
            min_entities=min_entities,
            q=q,
        )
        with Session(self.engine) as session:
            count_stmt = select(func.count()).select_from(Interaction)
            list_stmt = select(Interaction)
            if conds:
                count_stmt = count_stmt.where(*conds)
                list_stmt = list_stmt.where(*conds)
            total = session.exec(count_stmt).one()
            rows = session.exec(
                list_stmt.order_by(direction).offset((page - 1) * page_size).limit(page_size)
            ).all()
        return list(rows), int(total)

    def iter_interactions(
        self,
        *,
        limit: int | None = None,
        sources: Sequence[str] | None = None,
        wire_formats: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        policies: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        entity_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_entities: int | None = None,
        q: str | None = None,
    ) -> list[Interaction]:
        """All interactions matching the filters, newest first (for export)."""
        conds = self._conditions(
            sources=sources,
            wire_formats=wire_formats,
            models=models,
            policies=policies,
            languages=languages,
            entity_type=entity_type,
            date_from=date_from,
            date_to=date_to,
            min_entities=min_entities,
            q=q,
        )
        with Session(self.engine) as session:
            stmt = select(Interaction)
            if conds:
                stmt = stmt.where(*conds)
            stmt = stmt.order_by(col(Interaction.created_at).desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt).all())

    def get_interaction(self, interaction_id: int) -> Interaction | None:
        """Fetch a single interaction by id, or ``None`` if it doesn't exist."""
        with Session(self.engine) as session:
            return session.get(Interaction, interaction_id)

    def detections(self, interaction_id: int) -> list[Detection]:
        """Per-entity-type detection rows for one interaction, most frequent first."""
        with Session(self.engine) as session:
            statement = (
                select(Detection)
                .where(Detection.interaction_id == interaction_id)
                .order_by(col(Detection.count).desc())
            )
            return list(session.exec(statement).all())

    def text_counts_for(self, interaction_ids: Sequence[int]) -> dict[int, int]:
        """Map each given interaction id to how many text segments it has."""
        if not interaction_ids:
            return {}
        with Session(self.engine) as session:
            rows = session.exec(
                select(InteractionText.interaction_id, func.count())
                .where(col(InteractionText.interaction_id).in_(interaction_ids))
                .group_by(col(InteractionText.interaction_id))
            ).all()
        return {int(iid): int(n) for iid, n in rows}

    def distinct_values(self) -> dict[str, list[str]]:
        """Distinct filter values for the dashboard's dropdowns."""
        with Session(self.engine) as session:
            sources = session.exec(select(col(Interaction.source)).distinct()).all()
            wire_formats = session.exec(select(col(Interaction.wire_format)).distinct()).all()
            models = session.exec(select(col(Interaction.model)).distinct()).all()
            policies = session.exec(select(col(Interaction.policy)).distinct()).all()
            languages = session.exec(select(col(Interaction.language)).distinct()).all()
            entity_types = session.exec(select(col(Detection.entity_type)).distinct()).all()
        return {
            "sources": sorted(v for v in sources if v),
            "wire_formats": sorted(v for v in wire_formats if v),
            "models": sorted(v for v in models if v),
            "policies": sorted(v for v in policies if v),
            "languages": sorted(v for v in languages if v),
            "entity_types": sorted(v for v in entity_types if v),
        }

    def dashboard_summary(self) -> dict[str, Any]:
        """Rich aggregates for the overview page: totals, breakdowns, time series."""
        day = func.strftime("%Y-%m-%d", col(Interaction.created_at))
        with Session(self.engine) as session:
            interactions = int(session.exec(select(func.count()).select_from(Interaction)).one())
            by_type = {
                etype: int(total)
                for etype, total in session.exec(
                    select(Detection.entity_type, func.sum(Detection.count)).group_by(
                        Detection.entity_type
                    )
                ).all()
            }
            by_source = self._group_count(session, Interaction.source)
            by_wire_format = self._group_count(session, Interaction.wire_format)
            by_policy = self._group_count(session, Interaction.policy)
            by_model = self._group_count(session, Interaction.model)
            tokens_in = session.exec(
                select(func.coalesce(func.sum(col(Interaction.input_tokens)), 0))
            ).one()
            tokens_out = session.exec(
                select(func.coalesce(func.sum(col(Interaction.output_tokens)), 0))
            ).one()
            series_rows = session.exec(
                select(
                    day,
                    func.count(),
                    func.coalesce(func.sum(col(Interaction.entity_total)), 0),
                )
                .group_by(day)
                .order_by(day)
            ).all()
        timeseries = [
            {"date": str(d), "interactions": int(n), "entities": int(ents)}
            for d, n, ents in series_rows
        ]
        return {
            "interactions": interactions,
            "entities_total": sum(by_type.values()),
            "entities_by_type": by_type,
            "by_source": by_source,
            "by_wire_format": by_wire_format,
            "by_policy": by_policy,
            "by_model": by_model,
            "tokens": {"input": int(tokens_in or 0), "output": int(tokens_out or 0)},
            "timeseries": timeseries,
        }

    @staticmethod
    def _group_count(session: Session, column: Any) -> dict[str, int]:
        rows = session.exec(select(column, func.count()).group_by(column)).all()
        return {str(value): int(n) for value, n in rows if value is not None}

    # --- Management (used by the dashboard's write actions) ----------------

    def delete_interaction(self, interaction_id: int) -> bool:
        """Delete one interaction and its children. Returns False if it was absent."""
        with Session(self.engine) as session:
            if session.get(Interaction, interaction_id) is None:
                return False
            session.execute(
                delete(InteractionText).where(col(InteractionText.interaction_id) == interaction_id)
            )
            session.execute(
                delete(Detection).where(col(Detection.interaction_id) == interaction_id)
            )
            session.execute(delete(Interaction).where(col(Interaction.id) == interaction_id))
            session.commit()
            return True

    def clear_all(self) -> int:
        """Wipe every audit row. Returns how many interactions were removed."""
        with Session(self.engine) as session:
            removed = int(session.exec(select(func.count()).select_from(Interaction)).one())
            session.execute(delete(InteractionText))
            session.execute(delete(Detection))
            session.execute(delete(Interaction))
            session.commit()
            return removed
