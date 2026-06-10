"""Metadata-only audit store.

Records every interaction (timestamp, source tool, model, entity counts, token
stats) to SQLite. By design it NEVER stores raw PII; an encrypted raw-values
table may be added later behind config.
"""

from privacy_kit.gateway.store.audit import AuditStore
from privacy_kit.gateway.store.models import Detection, Interaction

__all__ = ["AuditStore", "Detection", "Interaction"]
