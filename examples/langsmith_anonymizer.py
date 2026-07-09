from __future__ import annotations

import os

from privacy_kit.integrations.langsmith import make_anonymizer, make_client

PAYLOAD_WITH_PII = {
    "input": "Jan Kowalski, jan.kowalski@example.com, +48 501 222 333",
    "output": "Wyślemy odpowiedź do Jana Kowalskiego.",
    "metadata": {
        "customer_email": "jan.kowalski@example.com",
        "trace_id": "demo-trace-001",
    },
}


def main() -> None:
    anonymizer = make_anonymizer()
    print("Local anonymized preview:")
    print(anonymizer(PAYLOAD_WITH_PII))

    if not os.getenv("LANGSMITH_API_KEY"):
        print("\nSet LANGSMITH_API_KEY to send this example to LangSmith.")
        return

    # make_client wires the anonymizer into inputs/outputs/errors and sets
    # hide_metadata=True (metadata is not routed through the anonymizer).
    client = make_client(api_key=os.getenv("LANGSMITH_API_KEY"))

    run = client.create_run(
        name="privacy-kit-langsmith-example",
        run_type="chain",
        inputs={"input": PAYLOAD_WITH_PII["input"]},
        outputs={"output": PAYLOAD_WITH_PII["output"]},
        extra={"metadata": PAYLOAD_WITH_PII["metadata"]},
    )
    client.flush()
    print(f"\nSent anonymized run to LangSmith: {run}")


if __name__ == "__main__":
    main()
