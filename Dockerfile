# privacy-kit gateway — local PII-filtering proxy for AI tools.
#
#   docker build -t privacy-kit .
#   docker run -p 127.0.0.1:8787:8787 -v privacy-kit-data:/data privacy-kit
#
# The model weights are baked into the image at build time, so `docker run`
# works offline and serves its first request warm. The audit DB lives on the
# /data volume so it survives container restarts. The detector is pure
# onnxruntime — no torch — so the image stays slim.

FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN uv venv "$VIRTUAL_ENV"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install ".[gateway]"

# Bake the model into the image (one HF download at build time, none at run time).
ENV HF_HOME=/opt/hf-cache
RUN python -c "from privacy_kit.core.detectors import BardsAiOnnxDetector; BardsAiOnnxDetector()"

# Inside a container the gateway must bind 0.0.0.0 to be reachable; publishing
# the port only to localhost (-p 127.0.0.1:8787:8787) keeps it machine-local.
ENV PII_HOST=0.0.0.0 \
    PII_DB_PATH=/data/privacy_kit.sqlite

RUN mkdir -p /data
VOLUME /data
EXPOSE 8787

CMD ["privacy-kit", "serve"]
