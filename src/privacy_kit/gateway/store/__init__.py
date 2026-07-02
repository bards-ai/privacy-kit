"""Audit store.

Records every interaction (timestamp, source tool, model, entity counts, token
stats) to SQLite, plus the request text segments selected by
``Settings.save_texts``.  Only user-authored text and tool/file data are
eligible for saving; system prompts, instruction blocks, tool-call arguments,
and assistant turns are never stored (though they are still anonymized before
being forwarded upstream).  Among the eligible segments, ``save_texts``
chooses which subset to persist: ``"anonymized"`` keeps only segments where PII
was detected and replaced; ``"all"`` keeps every eligible segment.
See ``InteractionText``.
"""

from privacy_kit.gateway.store.audit import AuditStore
from privacy_kit.gateway.store.models import Detection, Interaction, InteractionText

__all__ = ["AuditStore", "Detection", "Interaction", "InteractionText"]
