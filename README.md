# privacy-kit

PII detection and anonymization for multilingual text, powered by
[bardsai/eu-pii-anonimization-multilang](https://huggingface.co/bardsai/eu-pii-anonimization-multilang).

## Installation

```bash
pip install privacy-kit
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/bards-ai/privacy-kit.git
```

For development:

```bash
git clone https://github.com/bards-ai/privacy-kit.git
cd privacy-kit
uv venv && uv pip install -e .
```

## Quick start

```python
from privacy_kit import PiiModel

model = PiiModel().from_pretrained("bardsai/eu-pii-anonimization-multilang")

text = "Anna Wiśniewska mieszka na ul. Piękna 22, 00-549 Warszawa."

# Simple anonymization
model.anonymize(text)
# → '[PERSON_NAME] mieszka na [LOCATION].'

# Anonymization with entity IDs
model.anonymize(text, mode="ids")
# → {'anonymized_text': '[PERSON_NAME:1] mieszka na [LOCATION:1].',
#    'entities': {'PERSON_NAME:1': 'Anna Wiśniewska',
#                 'LOCATION:1': 'ul. Piękna 22, 00-549 Warszawa'}}

# Extract PII entities
model.extract_pii(text)
# → {'Anna Wiśniewska': 'PERSON_NAME',
#    'ul. Piękna 22, 00-549 Warszawa': 'LOCATION'}
```

## Supported entity types

The underlying model recognizes EU-relevant PII categories including:
`PERSON_NAME`, `LOCATION`, `FINANCIAL_AMOUNT`, `PERSON_IDENTIFIER`,
`ORGANIZATION_IDENTIFIER`, `PROPER_NAME`, `RELIGION_OR_BELIEF`,
`TRADE_UNION_MEMBERSHIP`, and more.

## License

Apache-2.0
