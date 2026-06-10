## Repo Notes

- Product direction: `privacy-kit` should feel like a kit of ready-to-use privacy blocks. Prefer import-use-done APIs over model-wrapper-only UX.
- Default install should stay lightweight and use the ONNX backend. Keep `torch` and `transformers` behind the optional `privacy-kit[transformers]` extra for the legacy `PiiModel` API.
- Observability integrations should default to `[REDACTED]`; legacy `PiiModel.anonymize()` can keep label placeholders like `[PERSON_NAME]`.
- Examples should live under `examples/` as repo onboarding assets, not installed package modules. Add examples only for integrations that are actually supported.
