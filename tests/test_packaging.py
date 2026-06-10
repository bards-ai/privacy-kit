import subprocess
import sys
from pathlib import Path


def test_importing_library_does_not_import_optional_dependencies() -> None:
    code = """
import sys
import privacy_kit
blocked = {'fastapi', 'uvicorn', 'httpx', 'langfuse', 'langchain', 'langgraph', 'torch', 'transformers'}
print(','.join(sorted(blocked & set(sys.modules))))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )

    assert result.stdout.strip() == ""
