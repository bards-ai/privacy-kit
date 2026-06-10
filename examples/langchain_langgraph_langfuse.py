from __future__ import annotations

import os

from langchain.agents import create_agent
from langfuse import get_client

from privacy_kit.integrations.langchain import make_langfuse_callback


PROMPT_WITH_PII = """
Rewrite this customer note as one concise status update.

Customer: Jan Kowalski
Email: jan.kowalski@example.com
Phone: +48 501 222 333
PESEL: 85010112345
Address: ul. Dluga 12, 00-001 Warszawa
""".strip()


def main() -> None:
    model = os.getenv("LANGCHAIN_MODEL")
    if model is None:
        model = "groq:llama-3.1-8b-instant" if os.getenv("GROQ_API_KEY") else "openai:gpt-4o-mini"

    langfuse_handler = make_langfuse_callback()
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You rewrite customer notes into short operational updates.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": PROMPT_WITH_PII}]},
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "demo": "privacy-kit-langchain-langgraph",
                "raw_contact": "jan.kowalski@example.com",
            },
        },
    )

    print(result["messages"][-1].content)
    get_client().flush()


if __name__ == "__main__":
    main()
