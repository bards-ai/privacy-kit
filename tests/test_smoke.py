"""Smoke test for privacy-kit API."""

import privacy_kit
from privacy_kit import Entity, PiiModel

SEP = "-" * 72

TEXTS = [
    "Pracownik Anna Wiśniewska, zatrudniona w dziale IT, otrzymuje wynagrodzenie "
    "w wysokości 12 500 PLN brutto. Numer paszportu: EP1234567.",
    "Jan Kowalski mieszka na ul. Piękna 22, 00-549 Warszawa. NIP: 527-020-1234.",
    "Cristiano Ronaldo scored a goal in Madrid.",
]


def main() -> None:
    print(f"privacy-kit {privacy_kit.__version__}")
    print(f"Entity class: {Entity}\n")

    print("Loading model…")
    model = PiiModel()
    print(f"  {model!r}\n")

    for text in TEXTS:
        print(SEP)
        print(f"INPUT:  {text}\n")

        pii = model.extract_pii(text)
        print(f"  extract_pii    → {pii}")

        simple = model.anonymize(text)
        print(f"  anonymize      → {simple}")

        ids = model.anonymize(text, mode="ids")
        print(f"  anonymize(ids) → {ids}")

        # --- assertions ---
        for entity_text in pii:
            assert entity_text == entity_text.strip(
                ".,;:!?"
            ), f"Entity has stray punctuation: {entity_text!r}"

        if pii:
            assert "[" in simple, "anonymize() should contain placeholders"

        if ids["entities"]:
            for key in ids["entities"]:
                label, num = key.rsplit(":", 1)
                assert num.isdigit(), f"ID part should be numeric: {key!r}"

        print("  ✓ assertions passed")
        print()

    # check numbering order (first entity in text should be :1)
    text = "Jan Kowalski i Anna Nowak mieszkają w Krakowie."
    ids = model.anonymize(text, mode="ids")
    keys = [k for k in ids["entities"] if k.startswith("PERSON_NAME:")]
    if len(keys) >= 2:
        assert ids["entities"]["PERSON_NAME:1"] == "Jan Kowalski", (
            f"First PERSON_NAME should be :1, got: {ids['entities']}"
        )
        print(f"{SEP}\nNumbeing order check: {ids}")
        print("  ✓ PERSON_NAME:1 is the first entity in text\n")

    # --- chunking test: text longer than max_length ---
    print(f"{SEP}\nChunking test (long text):\n")
    base = (
        "Jan Kowalski mieszka na ul. Piękna 22, 00-549 Warszawa. "
        "Anna Nowak pracuje w firmie przy ul. Mokotowskiej 10. "
    )
    # Repeat to exceed 512 subword tokens, then add a known entity at the end
    long_text = (base * 30) + "Cristiano Ronaldo gra w piłkę."
    word_count = len(long_text.split())
    print(f"  Word count: {word_count}")

    pii_long = model.extract_pii(long_text)
    entity_names = list(pii_long.keys())
    print(f"  Entities found: {len(pii_long)}")
    print(f"  Last entities: {dict(list(pii_long.items())[-3:])}")

    assert "Cristiano Ronaldo" in pii_long, (
        f"Entity at end of long text was lost! Found: {entity_names[-5:]}"
    )
    print("  ✓ Entity at end of long text preserved (chunking works)\n")

    print(f"{SEP}\nAll checks passed.")


if __name__ == "__main__":
    main()
