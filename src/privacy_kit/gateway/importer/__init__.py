"""Import past conversations from coding tools' on-disk history.

Parsers turn each tool's session transcript files into the neutral
:class:`~privacy_kit.gateway.importer.base.ParsedSession` shape; the runner
replays those through the same detect → anonymize → ``AuditStore.record``
pipeline the live proxy uses, so imported history renders in the dashboard
exactly like proxied traffic.
"""

from privacy_kit.gateway.importer.base import ParsedSession, ParsedTurn
from privacy_kit.gateway.importer.runner import ImportJob, run_import

__all__ = ["ImportJob", "ParsedSession", "ParsedTurn", "run_import"]
