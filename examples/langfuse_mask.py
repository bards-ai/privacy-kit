from __future__ import annotations

import os

from privacy_kit.integrations.langfuse import make_mask


PAYLOAD_WITH_PII = {
    "input": "Jan Kowalski, jan.kowalski@example.com, +48 501 222 333",
    "output": "Wyślemy odpowiedź do Jana Kowalskiego.",
    "metadata": {
        "customer_email": "jan.kowalski@example.com",
        "trace_id": "demo-trace-001",
    },
}


def main() -> None:
    mask = make_mask()
    print("Local masked preview:")
    print(mask(PAYLOAD_WITH_PII))

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("\nSet LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to send this example to Langfuse.")
        return

    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL"),
        mask=mask,
    )

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="privacy-kit-langfuse-example",
        model="demo-model",
        input=PAYLOAD_WITH_PII["input"],
        metadata=PAYLOAD_WITH_PII["metadata"],
    ) as generation:
        generation.update(output=PAYLOAD_WITH_PII["output"])
        trace_id = langfuse.get_current_trace_id()

    langfuse.flush()
    print(f"\nSent masked trace to Langfuse: {trace_id}")


if __name__ == "__main__":
    main()
