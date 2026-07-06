"""Audit store repository.

Thin wrapper over a SQLite engine for recording interactions and reading them
back — both the simple aggregates the legacy ``/ui`` needs and the richer
filtered, paginated queries the dashboard API (`/api/v1`) serves. Construct one
:class:`AuditStore` per process and reuse it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import ColumnElement, delete, event
from sqlmodel import Session, SQLModel, col, create_engine, func, select

from privacy_kit.gateway.config import get_settings
from privacy_kit.gateway.store.models import Detection, Interaction, InteractionText

# Imported history rows carry a "<source>-import" source (e.g.
# "claude-code-import", set by the importer parsers) so they can be told apart
# from live traffic. A filter on the live source name is meant to cover its
# imported sessions too, so base-source filters are expanded to match both.
IMPORT_SUFFIX = "-import"


def _expand_sources(sources: Sequence[str]) -> list[str]:
    """Add the ``-import`` variant of each requested source so that filtering by
    a live source (``claude-code``) also matches its imported rows
    (``claude-code-import``). Requesting the ``-import`` value directly stays
    exact."""
    expanded: list[str] = []
    for source in sources:
        expanded.append(source)
        if not source.endswith(IMPORT_SUFFIX):
            expanded.append(f"{source}{IMPORT_SUFFIX}")
    return expanded


# Columns the list endpoint is allowed to sort by (anything else falls back to
# created_at), keyed by the name the API accepts.
_SORT_COLUMNS = {
    "created_at": Interaction.created_at,
    "source": Interaction.source,
    "wire_format": Interaction.wire_format,
    "kind": Interaction.kind,
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
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Additively migrate older databases that predate newer columns.

        ``create_all`` only creates missing *tables*, never new columns on an
        existing one. Columns added to the model after a database was first
        created are patched in here with a default, so upgrading in place never
        requires a manual migration or a wipe. Each step is idempotent.
        """
        with self.engine.begin() as conn:
            cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(interaction)")}
            if "kind" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE interaction ADD COLUMN kind VARCHAR NOT NULL DEFAULT 'main'"
                )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_interaction_kind ON interaction (kind)"
            )
            if "conversation_id" not in cols:
                conn.exec_driver_sql("ALTER TABLE interaction ADD COLUMN conversation_id VARCHAR")
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_interaction_conversation_id "
                "ON interaction (conversation_id)"
            )
            text_cols = {
                row[1] for row in conn.exec_driver_sql("PRAGMA table_info(interactiontext)")
            }
            if "category" not in text_cols:
                conn.exec_driver_sql(
                    "ALTER TABLE interactiontext ADD COLUMN category VARCHAR NOT NULL "
                    "DEFAULT 'user'"
                )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_interactiontext_category "
                "ON interactiontext (category)"
            )

    def record(
        self,
        *,
        source: str,
        wire_format: str,
        model: str,
        entity_counts: Mapping[str, int],
        kind: str = "main",
        policy: str = "pseudonymize",
        language: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        conversation_id: str | None = None,
        created_at: datetime | None = None,
        texts: Sequence[tuple[str, str] | tuple[str, str, str]] = (),
    ) -> int:
        """Persist one interaction, its per-type detections, and any saved text
        segments, in capture order; return its id.

        Each ``texts`` entry is ``(original, anonymized)`` or, to record the
        segment's origin, ``(original, anonymized, category)`` where category is
        ``"user"`` (human-typed) or ``"tool"`` (tool/file data). Two-tuples default
        the category to ``"user"``.

        ``created_at`` backdates the row — importers pass the message's original
        timestamp so history sorts correctly; live traffic leaves it unset.
        """
        counts = {etype: int(n) for etype, n in entity_counts.items() if n}
        interaction = Interaction(
            source=source,
            wire_format=wire_format,
            kind=kind,
            model=model,
            policy=policy,
            language=language,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            entity_total=sum(counts.values()),
            entity_counts=counts,
            conversation_id=conversation_id,
        )
        if created_at is not None:
            interaction.created_at = created_at
        with Session(self.engine) as session:
            session.add(interaction)
            session.flush()  # assigns interaction.id
            assert interaction.id is not None
            for etype, count in counts.items():
                session.add(
                    Detection(interaction_id=interaction.id, entity_type=etype, count=count)
                )
            for seq, entry in enumerate(texts):
                original, anonymized = entry[0], entry[1]
                category = entry[2] if len(entry) > 2 else "user"
                session.add(
                    InteractionText(
                        interaction_id=interaction.id,
                        seq=seq,
                        category=category,
                        original=original,
                        anonymized=anonymized,
                    )
                )
            session.commit()
            return interaction.id

    def existing_conversation_ids(self, ids: Iterable[str]) -> set[str]:
        """Subset of ``ids`` that already have at least one interaction.

        Importers use this to skip whole sessions on re-run. Chunked so an
        arbitrarily large id list never exceeds SQLite's bound-parameter limit.
        """
        wanted = [i for i in ids if i]
        found: set[str] = set()
        with Session(self.engine) as session:
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                rows = session.exec(
                    select(Interaction.conversation_id)
                    .where(col(Interaction.conversation_id).in_(chunk))
                    .distinct()
                ).all()
                found.update(r for r in rows if r)
        return found

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
        kinds: Sequence[str] | None = None,
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
            conds.append(col(Interaction.source).in_(_expand_sources(sources)))
        if wire_formats:
            conds.append(col(Interaction.wire_format).in_(wire_formats))
        if kinds:
            conds.append(col(Interaction.kind).in_(kinds))
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
        kinds: Sequence[str] | None = None,
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
            kinds=kinds,
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
        kinds: Sequence[str] | None = None,
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
            kinds=kinds,
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

    # Sort keys the conversation list accepts, mapped to their aggregate
    # expression (anything else falls back to last_seen).
    def _conversation_sort(self, sort: str) -> Any:
        if sort == "first_seen":
            return func.min(col(Interaction.created_at))
        if sort == "turn_count":
            return func.count()
        if sort == "entity_total":
            return func.sum(col(Interaction.entity_total))
        return func.max(col(Interaction.created_at))  # "last_seen" (default)

    def list_conversations(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        sort: str = "last_seen",
        order: str = "desc",
        sources: Sequence[str] | None = None,
        wire_formats: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        policies: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        entity_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_entities: int | None = None,
        q: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paginated conversation summaries (grouped turns), plus total count.

        Groups interactions by ``conversation_id`` (NULL ids are ungrouped
        single-shot interactions and are excluded). Each returned dict carries:
        conversation_id, first_seen, last_seen, turn_count, entity_total,
        entity_counts (merged across turns), sources, models, background_count
        (non-main kind turns).
        """
        page = max(1, page)
        page_size = max(1, min(page_size, 200))

        # Row-level filters (same as the interaction list) plus the grouping
        # requirement that the row actually belongs to a conversation.
        conds = self._conditions(
            sources=sources,
            wire_formats=wire_formats,
            kinds=kinds,
            models=models,
            policies=policies,
            languages=languages,
            entity_type=entity_type,
            date_from=date_from,
            date_to=date_to,
            min_entities=min_entities,
            q=q,
        )
        conds.append(col(Interaction.conversation_id).is_not(None))

        sort_expr = self._conversation_sort(sort)
        direction = sort_expr.asc() if order == "asc" else sort_expr.desc()

        group_stmt = (
            select(col(Interaction.conversation_id))
            .where(*conds)
            .group_by(col(Interaction.conversation_id))
            # conversation_id tiebreaker keeps pagination deterministic when the
            # sort key ties across groups.
            .order_by(direction, col(Interaction.conversation_id).asc())
        )

        with Session(self.engine) as session:
            total = int(session.exec(select(func.count()).select_from(group_stmt.subquery())).one())
            # The is_not(None) filter above guarantees no NULL ids; the check
            # here only narrows the column's Optional type for mypy.
            conv_ids = [
                cid
                for cid in session.exec(
                    group_stmt.offset((page - 1) * page_size).limit(page_size)
                ).all()
                if cid is not None
            ]

            # Fetch every turn of the page's conversations in one query, then
            # fold per conversation in Python (turn counts are small).
            result: list[dict[str, Any]] = []
            if conv_ids:
                rows = session.exec(
                    select(Interaction)
                    .where(col(Interaction.conversation_id).in_(conv_ids))
                    .order_by(col(Interaction.created_at).asc())
                ).all()
                by_conv: dict[str, list[Interaction]] = {cid: [] for cid in conv_ids}
                for row in rows:
                    if row.conversation_id is not None:
                        by_conv[row.conversation_id].append(row)
                # Preserve the sorted/paginated order from group_stmt.
                for cid in conv_ids:
                    result.append(self.conversation_summary(cid, by_conv[cid]))

        return result, total

    @staticmethod
    def conversation_summary(
        conversation_id: str, interactions: Sequence[Interaction]
    ) -> dict[str, Any]:
        """Roll a conversation's turns up into a single summary dict."""
        created = [i.created_at for i in interactions]
        merged_counts: dict[str, int] = {}
        for interaction in interactions:
            for etype, count in interaction.entity_counts.items():
                merged_counts[etype] = merged_counts.get(etype, 0) + count
        return {
            "conversation_id": conversation_id,
            "first_seen": min(created).isoformat() if created else None,
            "last_seen": max(created).isoformat() if created else None,
            "turn_count": len(interactions),
            "entity_total": sum(i.entity_total for i in interactions),
            "entity_counts": merged_counts,
            "sources": sorted({i.source for i in interactions}),
            "models": sorted({i.model for i in interactions}),
            "background_count": sum(1 for i in interactions if i.kind != "main"),
            "input_tokens": sum(i.input_tokens or 0 for i in interactions),
            "output_tokens": sum(i.output_tokens or 0 for i in interactions),
        }

    def get_conversation(self, conversation_id: str) -> list[Interaction] | None:
        """Fetch a conversation's turns (Interaction rows) ordered oldest-first.

        Returns ``None`` when the conversation has no turns. Callers compose the
        response (detections/texts, plaintext redaction) via the same helpers
        the single-interaction detail endpoint uses.
        """
        with Session(self.engine) as session:
            interactions = list(
                session.exec(
                    select(Interaction)
                    .where(col(Interaction.conversation_id) == conversation_id)
                    .order_by(col(Interaction.created_at).asc())
                ).all()
            )
        return interactions or None

    def turn_had_pii(self, conversation_id: str) -> bool:
        """Return True if the most recent exchange in this cursor conversation detected PII.

        "Most recent exchange" is the run of cursor-source interactions (ordered
        newest-first) that precede the last interaction already associated with an
        ``assistant``-category text segment — i.e. the turns since the previous
        response was recorded. This mirrors the main proxy's ``count_vault.type_counts``
        check that gates response capture on the *prompt's* PII, not the reply's.

        Returns ``False`` when the conversation has no interactions or when the
        look-up fails, so callers safely treat the result as "don't save" in those
        edge cases.
        """
        with Session(self.engine) as session:
            interactions = list(
                session.exec(
                    select(Interaction)
                    .where(
                        col(Interaction.conversation_id) == conversation_id,
                        col(Interaction.source) == "cursor",
                    )
                    .order_by(col(Interaction.created_at).desc())
                ).all()
            )
            if not interactions:
                return False
            # Walk newest-first; stop as soon as we hit an interaction that
            # already has an assistant text row (= previous exchange's response).
            # The trailing window up to (not including) that stop point is the
            # current exchange's prompt/file-read turns.
            current_exchange: list[Interaction] = []
            for interaction in interactions:
                has_assistant = session.exec(
                    select(InteractionText.id)
                    .where(
                        col(InteractionText.interaction_id) == interaction.id,
                        col(InteractionText.category) == "assistant",
                    )
                    .limit(1)
                ).first()
                if has_assistant is not None:
                    break
                current_exchange.append(interaction)
        return any(i.entity_total > 0 for i in current_exchange)

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
            kinds = session.exec(select(col(Interaction.kind)).distinct()).all()
            models = session.exec(select(col(Interaction.model)).distinct()).all()
            policies = session.exec(select(col(Interaction.policy)).distinct()).all()
            languages = session.exec(select(col(Interaction.language)).distinct()).all()
            entity_types = session.exec(select(col(Detection.entity_type)).distinct()).all()
        return {
            "sources": sorted(v for v in sources if v),
            "wire_formats": sorted(v for v in wire_formats if v),
            "kinds": sorted(v for v in kinds if v),
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
